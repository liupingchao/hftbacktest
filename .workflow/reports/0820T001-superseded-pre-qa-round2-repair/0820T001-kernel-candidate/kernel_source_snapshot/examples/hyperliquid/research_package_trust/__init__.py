"""Research Package Trust Kernel v1 candidate public API."""

from .admission import AdmissionResult, admit_package
from .canonical import (
    canonical_json_bytes,
    canonical_json_sha256,
    canonical_pretty_json_bytes,
    read_json,
    read_json_object,
    sha256_bytes,
    sha256_file,
)
from .contracts import (
    get_accepted_version,
    load_accepted_version_registry,
    validate_accepted_version_package,
    validate_json_schema,
    validate_pinned_version,
    validate_registry_append_only,
)
from .errors import TrustKernelError
from .evidence import validate_exact_object
from .identity import (
    PackageIdentity,
    build_package_identity,
    compute_composite_package_identity,
    compute_publication_envelope_identity,
    compute_research_data_identity,
    compute_runtime_contract_identity,
    validate_identity_bindings,
)
from .operations import (
    assert_archive_tree_match,
    capture_cleanup_preflight,
    recheck_cleanup_preflight,
)
from .publication import (
    assert_zero_write_snapshot,
    atomic_write_bytes,
    atomic_write_json,
    fsync_tree,
    metadata_snapshot,
    publish_atomically,
)
from .tree import (
    TreeEntry,
    build_inventory,
    scan_exact_tree,
    tree_snapshot,
    validate_inventory_against_surface,
    validate_relative_path,
)

__all__ = [
    "AdmissionResult",
    "PackageIdentity",
    "TreeEntry",
    "TrustKernelError",
    "admit_package",
    "assert_archive_tree_match",
    "assert_zero_write_snapshot",
    "atomic_write_bytes",
    "atomic_write_json",
    "build_inventory",
    "build_package_identity",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "canonical_pretty_json_bytes",
    "capture_cleanup_preflight",
    "compute_composite_package_identity",
    "compute_publication_envelope_identity",
    "compute_research_data_identity",
    "compute_runtime_contract_identity",
    "fsync_tree",
    "get_accepted_version",
    "load_accepted_version_registry",
    "metadata_snapshot",
    "publish_atomically",
    "read_json",
    "read_json_object",
    "recheck_cleanup_preflight",
    "scan_exact_tree",
    "sha256_bytes",
    "sha256_file",
    "tree_snapshot",
    "validate_exact_object",
    "validate_accepted_version_package",
    "validate_identity_bindings",
    "validate_inventory_against_surface",
    "validate_json_schema",
    "validate_pinned_version",
    "validate_registry_append_only",
    "validate_relative_path",
]
