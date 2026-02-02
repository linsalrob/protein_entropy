"""
Configuration for pytest.
"""

import os


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that use real models",
    )


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test (requires real models)",
    )
    
    # Check for environment variable
    if os.environ.get("RUN_INTEGRATION") == "1":
        config.option.run_integration = True


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    if config.getoption("--run-integration"):
        # If --run-integration is given, don't skip integration tests
        return
    
    skip_integration = pytest.mark.skip(reason="need --run-integration option to run")
    
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
