"""Regression coverage: ``export_memory_md`` must not silently write an
empty export when every ``recall`` call fails (e.g. the on-prem backend is
unreachable), and ``sync_memory_to_project`` must fall back to a
previous good export instead of overwriting a project's ``MEMORY.md`` with
nothing.

Before this fix, a total-outage export produced an all-empty
``memories_by_type`` (each per-type ``recall`` exception was swallowed) and
still wrote it out — every call to ``memanto memory sync`` during a brief
backend outage silently wiped the agent's exported context and the
project's ``MEMORY.md``.
"""

from unittest.mock import MagicMock

import pytest

import memanto.cli.client.direct_client as direct_mod
import memanto.cli.client.sdk_client as sdk_mod
from memanto.app.services.memory_export_service import MEMORY_TYPE_ORDER

DirectClient = direct_mod.DirectClient
SdkClient = sdk_mod.SdkClient


def _build_client(client_cls, monkeypatch, tmp_path):
    """Construct *client_cls* with session validation stubbed out and
    ``Path.home()`` redirected to *tmp_path*. ``Path`` is the same class
    object everywhere it's imported, so this one patch also covers
    ``MemoryExportService``'s default ``exports_dir`` — export writes and
    ``sync_memory_to_project``'s cache lookup end up at the same
    ``tmp_path/.memanto/exports/`` regardless of which module reads
    ``Path.home()``."""
    module = direct_mod if client_cls is DirectClient else sdk_mod
    monkeypatch.setattr(module.Path, "home", classmethod(lambda cls: tmp_path))

    client = client_cls(api_key="test-key")
    monkeypatch.setattr(
        client, "_get_validated_session_for_agent", lambda agent_id: MagicMock(namespace="test-namespace")
    )
    return client


class TestExportMemoryMdRefusesEmptyOnTotalFailure:
    @pytest.mark.parametrize("client_cls", [SdkClient, DirectClient])
    def test_raises_when_every_recall_fails(self, client_cls, monkeypatch, tmp_path):
        client = _build_client(client_cls, monkeypatch, tmp_path)
        if client_cls is SdkClient:
            mock_moorcheh = MagicMock()
            mock_moorcheh.documents.fetch_text_data.side_effect = ConnectionError("backend down")
            monkeypatch.setattr(client, "_get_moorcheh", lambda: mock_moorcheh)
            expected_match = "complete memory set"
        else:
            monkeypatch.setattr(
                client, "recall", MagicMock(side_effect=ConnectionError("backend down"))
            )
            expected_match = "unreachable"

        with pytest.raises(ConnectionError, match=expected_match):
            client.export_memory_md(agent_id="test-agent")

    @pytest.mark.parametrize("client_cls", [SdkClient, DirectClient])
    def test_partial_failure_refuses_incomplete_export(
        self, client_cls, monkeypatch, tmp_path
    ):
        """One failed type must not be represented as a genuinely empty type."""
        client = _build_client(client_cls, monkeypatch, tmp_path)

        if client_cls is SdkClient:
            def fake_fetch(*args, **kwargs):
                if kwargs.get("next_token") == "token2":
                    raise ConnectionError("flaky")
                return {"items": [{"content": "ok", "type": MEMORY_TYPE_ORDER[0]}], "pagination": {"has_more": True, "next_token": "token2"}}
            mock_moorcheh = MagicMock()
            mock_moorcheh.documents.fetch_text_data.side_effect = fake_fetch
            monkeypatch.setattr(client, "_get_moorcheh", lambda: mock_moorcheh)
            expected_match = "complete memory set"
        else:
            def fake_recall(agent_id, query, limit, type):
                if type == [MEMORY_TYPE_ORDER[0]]:
                    raise ConnectionError("flaky")
                return {"memories": [{"content": "ok"}]}

            monkeypatch.setattr(client, "recall", MagicMock(side_effect=fake_recall))
            expected_match = f"incomplete.*{MEMORY_TYPE_ORDER[0]}|{MEMORY_TYPE_ORDER[0]}.*incomplete"

        with pytest.raises(
            ConnectionError,
            match=expected_match,
        ):
            client.export_memory_md(agent_id="test-agent")


class TestSyncFallsBackToCache:
    """Sync refreshes first, so a cached export is only reused when the
    refresh fails — never in place of memories written this session."""

    @pytest.mark.parametrize("client_cls", [SdkClient, DirectClient])
    def test_cache_used_when_backend_down(self, client_cls, monkeypatch, tmp_path):
        client = _build_client(client_cls, monkeypatch, tmp_path)

        cache_file = tmp_path / ".memanto" / "exports" / "test-agent_memory.md"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text("### Some Memory\n\ngood content\n", encoding="utf-8")

        if client_cls is SdkClient:
            mock_moorcheh = MagicMock()
            mock_moorcheh.documents.fetch_text_data.side_effect = ConnectionError("backend down")
            monkeypatch.setattr(client, "_get_moorcheh", lambda: mock_moorcheh)
        else:
            monkeypatch.setattr(
                client, "recall", MagicMock(side_effect=ConnectionError("backend down"))
            )

        project_dir = tmp_path / "project"
        result = client.sync_memory_to_project(
            agent_id="test-agent", project_dir=str(project_dir)
        )

        if client_cls is DirectClient:
            client.recall.assert_called()
        assert result["source"] == "stale-cache"
        assert result["total_memories"] == 1
        written = (project_dir / "MEMORY.md").read_text(encoding="utf-8")
        assert "good content" in written

    @pytest.mark.parametrize("client_cls", [SdkClient, DirectClient])
    def test_fresh_export_replaces_stale_cache(self, client_cls, monkeypatch, tmp_path):
        """A cache written before this session must not shadow new memories."""
        client = _build_client(client_cls, monkeypatch, tmp_path)

        cache_file = tmp_path / ".memanto" / "exports" / "test-agent_memory.md"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text("### Old Memory\n\nstale content\n", encoding="utf-8")

        if client_cls is SdkClient:
            mock_moorcheh = MagicMock()
            mock_moorcheh.documents.fetch_text_data.return_value = {"items": [{"content": "fresh content", "type": "instruction"}], "pagination": {"has_more": False}}
            monkeypatch.setattr(client, "_get_moorcheh", lambda: mock_moorcheh)
            mock_reader = MagicMock()
            mock_reader._format_memory_item.side_effect = lambda x: {"content": x.get("content"), "type": x.get("type")}
            monkeypatch.setattr(client, "_get_read_service", lambda: mock_reader)
        else:
            monkeypatch.setattr(
                client,
                "recall",
                MagicMock(return_value={"memories": [{"content": "fresh content"}]}),
            )

        project_dir = tmp_path / "project"
        result = client.sync_memory_to_project(
            agent_id="test-agent", project_dir=str(project_dir)
        )

        assert result["source"] == "fresh"
        written = (project_dir / "MEMORY.md").read_text(encoding="utf-8")
        assert "stale content" not in written
        assert "fresh content" in written

    def test_raises_when_no_cache_and_backend_down(self, monkeypatch, tmp_path):
        client = _build_client(SdkClient, monkeypatch, tmp_path)
        mock_moorcheh = MagicMock()
        mock_moorcheh.documents.fetch_text_data.side_effect = ConnectionError("backend down")
        monkeypatch.setattr(client, "_get_moorcheh", lambda: mock_moorcheh)

        with pytest.raises(ConnectionError):
            client.sync_memory_to_project(
                agent_id="test-agent", project_dir=str(tmp_path / "project")
            )

    @pytest.mark.parametrize("client_cls", [SdkClient, DirectClient])
    def test_rejects_path_traversal_before_cache_lookup(
        self, client_cls, monkeypatch, tmp_path
    ):
        client = _build_client(client_cls, monkeypatch, tmp_path)

        # Patch get_data_dir in the specific client module to prove it's never reached
        mock_get_data_dir = MagicMock()
        monkeypatch.setattr(f"{client_cls.__module__}.get_data_dir", mock_get_data_dir)

        with pytest.raises(ValueError, match="invalid characters"):
            client.sync_memory_to_project(
                agent_id="../outside", project_dir=str(tmp_path / "project")
            )

        mock_get_data_dir.assert_not_called()
