from __future__ import annotations

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry._logs import set_logger_provider, get_logger_provider, SeverityNumber
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter

from litter_detector.config import Settings


def setup_telemetry(service_name: str) -> None:
    """Initialize OTel providers for traces, metrics, and logs.

    Call once at process startup before any instrumented code runs.
    """
    settings = Settings()
    endpoint = settings.otel_endpoint
    resource = Resource.create({SERVICE_NAME: service_name})

    # Traces
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
    )
    trace.set_tracer_provider(tracer_provider)

    # Metrics
    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=endpoint, insecure=True),
        export_interval_millis=10_000,
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    # Logs
    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(
        BatchLogRecordProcessor(OTLPLogExporter(endpoint=endpoint, insecure=True))
    )
    set_logger_provider(logger_provider)

    # Bridge loguru to OTel
    _setup_loguru_bridge(logger_provider, service_name)


def shutdown_telemetry() -> None:
    """Flush and shut down all OTel providers."""
    provider = trace.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        provider.shutdown()

    meter_provider = metrics.get_meter_provider()
    if isinstance(meter_provider, MeterProvider):
        meter_provider.shutdown()

    log_provider = get_logger_provider()
    if isinstance(log_provider, LoggerProvider):
        log_provider.shutdown()


# -- Loguru bridge -------------------------------------------------------------

_LOGURU_TO_SEVERITY: dict[str, SeverityNumber] = {
    "TRACE": SeverityNumber.TRACE,
    "DEBUG": SeverityNumber.DEBUG,
    "INFO": SeverityNumber.INFO,
    "SUCCESS": SeverityNumber.INFO2,
    "WARNING": SeverityNumber.WARN,
    "ERROR": SeverityNumber.ERROR,
    "CRITICAL": SeverityNumber.FATAL,
}


def _setup_loguru_bridge(logger_provider: LoggerProvider, service_name: str) -> None:
    """Add a loguru sink that forwards log records to OTel."""
    from loguru import logger as loguru_logger

    otel_logger = logger_provider.get_logger(service_name)

    def _otel_sink(message) -> None:  # noqa: ANN001
        record = message.record
        severity = _LOGURU_TO_SEVERITY.get(
            record["level"].name, SeverityNumber.UNSPECIFIED
        )
        otel_logger.emit(
            timestamp=int(record["time"].timestamp() * 1e9),
            severity_number=severity,
            severity_text=record["level"].name,
            body=str(record["message"]),
            attributes={
                "logger.name": record["name"] or "litter_detector",
                "code.filepath": str(record["file"].path),
                "code.lineno": record["line"],
                "code.function": record["function"],
            },
        )

    loguru_logger.add(_otel_sink, level="DEBUG")
