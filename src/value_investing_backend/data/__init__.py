"""Point-in-time data access helpers for Rule One analysis."""

from value_investing_backend.data.edgar import EdgarClient
from value_investing_backend.data.insider_trades import (
    InsiderAlignmentResult,
    InsiderTransaction,
    fetch_insider_history,
    parse_form4_xml,
    summarize_insider_alignment,
)
from value_investing_backend.data.management_documents import (
    ArchiveBackedManagementProvider,
    ArchivedManagementDocument,
    DeterministicDiscoverer,
    DocumentCandidate,
    DocumentValidationPolicy,
    EdgarDiscoverer,
    FetchedManagementDocument,
    IRArchiveDiscoverer,
    IRSourcesConfig,
    IRSourcesEntry,
    LocalManagementDocumentArchive,
    ManagementDocumentFetcher,
    SupabaseManagementDocumentArchive,
    SupabaseManagementDocumentStore,
    bytes_to_text,
    default_management_archive_provider,
)
from value_investing_backend.data.pit_facts import FactValue, PointInTimeFacts
from value_investing_backend.data.prices import PriceClient
from value_investing_backend.data.transcripts import (
    DefaultTranscriptProvider,
    EarningsTranscript,
    FmpTranscriptProvider,
    Sec8KTranscriptProvider,
    TranscriptProvider,
)
from value_investing_backend.data.universe import SP500Universe

__all__ = [
    "DefaultTranscriptProvider",
    "ArchiveBackedManagementProvider",
    "ArchivedManagementDocument",
    "DeterministicDiscoverer",
    "DocumentCandidate",
    "DocumentValidationPolicy",
    "EarningsTranscript",
    "EdgarClient",
    "EdgarDiscoverer",
    "FetchedManagementDocument",
    "FactValue",
    "FmpTranscriptProvider",
    "IRArchiveDiscoverer",
    "IRSourcesConfig",
    "IRSourcesEntry",
    "InsiderAlignmentResult",
    "InsiderTransaction",
    "LocalManagementDocumentArchive",
    "ManagementDocumentFetcher",
    "PointInTimeFacts",
    "PriceClient",
    "SP500Universe",
    "Sec8KTranscriptProvider",
    "SupabaseManagementDocumentArchive",
    "SupabaseManagementDocumentStore",
    "TranscriptProvider",
    "bytes_to_text",
    "default_management_archive_provider",
    "fetch_insider_history",
    "parse_form4_xml",
    "summarize_insider_alignment",
]
