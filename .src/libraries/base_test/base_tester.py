import datetime
import json
import logging
import pathlib
import re
import typing
import uuid


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.datetime.fromtimestamp(
                record.created, datetime.timezone.utc
            ).isoformat(),
            "event": record.getMessage(),
            "metadata": getattr(record, "metadata", {}),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True, allow_nan=False)


class _JsonFileHandler(logging.Handler):
    def __init__(self, path: pathlib.Path) -> None:
        super().__init__()
        self.path = path
        self.records: typing.List[typing.Dict[str, typing.Any]] = []
        self._written = False

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(json.loads(self.format(record)))

    def close(self) -> None:
        if not self._written:
            with self.path.open("w", encoding="utf-8") as output:
                json.dump(
                    self.records, output, indent=2, sort_keys=True, allow_nan=False
                )
                output.write("\n")
            self._written = True
        super().close()


class _MetadataLogger(logging.LoggerAdapter):
    """Logger adapter that attaches JSON metadata to each record.

    Test implementations normally add fields through :meth:`add_metadata`,
    then write events with the standard logging methods::

        self._logger.add_metadata(model="my-model", batch_size=4)
        self._logger.info("model_loaded")

    Each emitted record includes those fields in its ``metadata`` object.
    Metadata values must be JSON-serializable.
    """

    def __init__(
        self, logger: logging.Logger, metadata: typing.Dict[str, typing.Any]
    ) -> None:
        super().__init__(logger, {})
        self._metadata = metadata

    def set_metadata(self, metadata: typing.Dict[str, typing.Any]) -> None:
        """Replace the current metadata fields.

        This is used by the base tester to initialize metadata at the start of
        a run. Test implementations should usually use :meth:`add_metadata`
        instead, so they preserve the base fields.
        """
        self._metadata = metadata

    def add_metadata(self, **metadata: typing.Any) -> None:
        """Add or replace JSON metadata fields for subsequent log records.

        Pass keyword arguments whose values can be encoded as JSON. New values
        replace existing values with the same key; all other metadata fields
        are retained.

        Example::

            self._logger.add_metadata(
                model="my-model",
                batch_size=4,
                use_gpu=True,
            )
            self._logger.info("benchmark_started")

        The resulting log record includes ``model``, ``batch_size``, and
        ``use_gpu`` alongside base metadata such as the test name and start
        time. Non-JSON-serializable values raise ``TypeError`` (or
        ``ValueError`` for unsupported numeric values) before metadata changes.
        """
        updated_metadata = {**self._metadata, **metadata}
        json.dumps(updated_metadata, allow_nan=False)
        self._metadata.update(metadata)

    def process(
        self, msg: typing.Any, kwargs: typing.MutableMapping[str, typing.Any]
    ) -> typing.Tuple[typing.Any, typing.MutableMapping[str, typing.Any]]:
        """Attach current metadata to a standard logging call."""
        extra = kwargs.setdefault("extra", {})
        if not isinstance(extra, typing.MutableMapping):
            raise TypeError("Logger extra must be a mutable mapping.")
        supplied_metadata = extra.get("metadata", {})
        if not isinstance(supplied_metadata, typing.Mapping):
            raise TypeError("Logger metadata must be a mapping.")
        extra["metadata"] = {**self._metadata, **supplied_metadata}
        return msg, kwargs


class base_tester:
    def __init__(self, name: str, log_dir: typing.Union[str, pathlib.Path] = ".dpl/logs"):
        self.name = name
        self.log_dir = pathlib.Path(log_dir)
        logger = logging.Logger(f"{__name__}.{id(self)}", logging.INFO)
        logger.propagate = False
        self._logger = _MetadataLogger(
            logger,
            {"test_name": self.name},
        )

    def start_test(self) -> typing.Dict[str, typing.Any]:
        """
        Initialize logging, run the subclass test, and write its JSON metadata.
        """
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now(datetime.timezone.utc)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.name).strip("._") or "test"
        log_path = self.log_dir / (
            f"{safe_name}_{timestamp:%Y%m%dT%H%M%S.%fZ}_{uuid.uuid4().hex}.json"
        )
        logger = self._logger.logger
        handler = _JsonFileHandler(log_path)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        self._logger.set_metadata(
            {
                "test_name": self.name,
                "started_at": timestamp.isoformat(),
            }
        )

        try:
            results = self._run_test()
        except Exception as error:
            self._logger.add_metadata(
                status="failed",
                error_type=type(error).__name__,
                error=str(error),
            )
            self._logger.info("test_finished")
            raise
        else:
            self._logger.add_metadata(status="completed")
            self._logger.info("test_finished")
            return results
        finally:
            logger.removeHandler(handler)
            handler.close()

    def _run_test(self) -> typing.Dict[str, typing.Any]:
        """Implement the test and return its results.

        Subclasses can add model-specific metadata while running::

            class ExampleTester(base_tester):
                def _run_test(self):
                    self._logger.add_metadata(model="example", backend="gpu")
                    self._logger.info("benchmark_started")
                    return {"tokens_per_second": 42.0}
        """
        raise NotImplementedError("Subclasses should implement this method.")