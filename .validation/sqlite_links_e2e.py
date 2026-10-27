from contextlib import closing
import importlib.metadata
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time

from google.adk.telemetry.sqlite_span_exporter import SqliteSpanExporter
import google.adk.telemetry.sqlite_span_exporter as exporter_module
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import Link, SpanContext, TraceFlags, TraceState

LINK_CONTEXT = SpanContext(
    trace_id=0x123456789abcdef0123456789abcdef0,
    span_id=0x123456789abcdef0,
    is_remote=True,
    trace_flags=TraceFlags(3),
    trace_state=TraceState([("vendor", "upstream"), ("other", "state")]),
)


def write(db):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(SqliteSpanExporter(db_path=db)))
    with provider.get_tracer("sqlite-links-smoke").start_as_current_span(
        "linked-operation",
        attributes={"gcp.vertex.agent.session_id": "session"},
        links=[Link(LINK_CONTEXT, {"reason": "retry", "attempt": 2})],
    ):
        pass
    provider.shutdown()


def read(db, expected_links):
    exporter = SqliteSpanExporter(db_path=db)
    try:
        spans = exporter.get_all_spans_for_session("session")
    finally:
        exporter.shutdown()
    linked = next(span for span in spans if span.name == "linked-operation")
    assert len(linked.links) == expected_links, len(linked.links)
    if expected_links:
        link = linked.links[0]
        assert link.context == LINK_CONTEXT, link.context
        assert link.attributes == {"reason": "retry", "attempt": 2}, link.attributes
    legacy = [span for span in spans if span.name == "legacy"]
    if legacy:
        assert not legacy[0].links
    print(json.dumps({"links": len(linked.links), "legacy_rows": len(legacy)}))


def main():
    expected_links = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    results = {}
    with tempfile.TemporaryDirectory() as directory:
        for name in ("fresh", "legacy"):
            db = str(Path(directory) / (name + ".db"))
            if name == "legacy":
                with sqlite3.connect(db) as conn:
                    conn.execute(
                        "CREATE TABLE spans (span_id TEXT PRIMARY KEY, trace_id TEXT NOT NULL,"
                        " parent_span_id TEXT, name TEXT NOT NULL, start_time_unix_nano INTEGER,"
                        " end_time_unix_nano INTEGER, session_id TEXT, invocation_id TEXT,"
                        " attributes_json TEXT)"
                    )
                    conn.execute(
                        "INSERT INTO spans (span_id, trace_id, name, session_id)"
                        " VALUES ('111', '222', 'legacy', 'session')"
                    )
                conn.close()
            subprocess.run([sys.executable, __file__, "write", db], check=True)
            output = subprocess.check_output(
                [sys.executable, __file__, "read", db, str(expected_links)], text=True
            )
            results[name] = json.loads(output)
            with closing(sqlite3.connect(db)) as writer:
                assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
                writer.execute("BEGIN IMMEDIATE")
                writer.execute("UPDATE spans SET name = 'uncommitted'")
                started = time.monotonic()
                try:
                    output = subprocess.check_output(
                        [sys.executable, __file__, "read", db, str(expected_links)],
                        text=True,
                    )
                    results[name]["wal_reader"] = json.loads(output)
                    results[name]["wal_readback_seconds"] = round(time.monotonic() - started, 4)
                finally:
                    writer.rollback()
    print(json.dumps({
        "python": sys.version.split()[0],
        "opentelemetry_sdk": importlib.metadata.version("opentelemetry-sdk"),
        "exporter": exporter_module.__file__,
        "separate_writer_reader_processes": True,
        "expected_links": expected_links,
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "write":
        write(sys.argv[2])
    elif len(sys.argv) > 1 and sys.argv[1] == "read":
        read(sys.argv[2], int(sys.argv[3]))
    else:
        main()
