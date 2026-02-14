#!/usr/bin/env python3
"""
Tests for Model Rollback Module
===============================

Tests model versioning, registry, and rollback functionality.
Aligned with the actual API in src/model_rollback.py.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestModelVersion:
    """Tests for ModelVersion dataclass."""

    def test_creation(self):
        """Test model version creation."""
        from model_rollback import ModelVersion

        version = ModelVersion(
            version='1.0.0',
            path='/models/v1/model.keras',
            created_at=datetime.now().isoformat(),
            deployed_at=None,
            metrics={'f1': 0.89, 'accuracy': 0.91},
            config_hash='abc123',
            data_version=None,
            status='archived',
        )

        assert version.version == '1.0.0'
        assert version.status == 'archived'
        assert version.metrics['f1'] == 0.89

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from model_rollback import ModelVersion

        version = ModelVersion(
            version='1.0.0',
            path='/models/v1/model.keras',
            created_at='2026-01-30T10:00:00',
            deployed_at=None,
            metrics={'f1': 0.89},
            config_hash='abc123',
            data_version=None,
            status='archived',
        )

        d = version.to_dict()

        assert isinstance(d, dict)
        assert d['version'] == '1.0.0'
        assert isinstance(d['path'], str)

    def test_from_dict_roundtrip(self):
        """Test from_dict(to_dict()) roundtrip."""
        from model_rollback import ModelVersion

        original = ModelVersion(
            version='2.0.0',
            path='/models/v2/model.keras',
            created_at='2026-02-01T12:00:00',
            deployed_at='2026-02-01T13:00:00',
            metrics={'accuracy': 0.92},
            config_hash='def456',
            data_version='v3',
            status='deployed',
        )

        restored = ModelVersion.from_dict(original.to_dict())

        assert restored.version == original.version
        assert restored.metrics == original.metrics
        assert restored.status == original.status


class TestModelRegistry:
    """Tests for ModelRegistry class."""

    def test_registry_initialization(self):
        """Test registry initialization creates directory."""
        from model_rollback import ModelRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            registry_dir = Path(tmpdir) / 'registry'
            registry = ModelRegistry(registry_dir=registry_dir)

            assert registry.registry_dir.exists()
            assert registry.registry['models'] == {}
            assert registry.registry['current_version'] is None

    def test_register_model(self):
        """Test model registration."""
        from model_rollback import ModelRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_dir=Path(tmpdir) / 'registry')

            # Create a mock model file
            model_path = Path(tmpdir) / 'new_model.keras'
            model_path.write_text('mock model data')

            version = registry.register_model(
                model_path=model_path,
                version='1.0.0',
                metrics={'f1': 0.90, 'accuracy': 0.91},
            )

            assert version.version == '1.0.0'
            assert version.metrics['f1'] == 0.90
            assert '1.0.0' in registry.registry['models']

    def test_deploy_model(self):
        """Test deploying a registered model."""
        from model_rollback import ModelRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_dir=Path(tmpdir) / 'registry')

            model_path = Path(tmpdir) / 'model.keras'
            model_path.write_text('mock')

            registry.register_model(
                model_path=model_path,
                version='1.0.0',
                metrics={'f1': 0.85},
            )

            success = registry.deploy_model('1.0.0')

            assert success is True
            assert registry.get_current_version() == '1.0.0'
            assert registry.registry['models']['1.0.0']['status'] == 'deployed'

    def test_deploy_nonexistent_version(self):
        """Test deploying a version that doesn't exist."""
        from model_rollback import ModelRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_dir=Path(tmpdir) / 'registry')
            success = registry.deploy_model('999.0.0')
            assert success is False

    def test_list_versions(self):
        """Test listing all versions."""
        from model_rollback import ModelRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_dir=Path(tmpdir) / 'registry')

            for i in range(3):
                model_path = Path(tmpdir) / f'model_{i}.keras'
                model_path.write_text(f'mock {i}')
                registry.register_model(
                    model_path=model_path,
                    version=f'{i+1}.0.0',
                    metrics={'f1': 0.80 + i * 0.05},
                )

            versions = registry.list_versions()
            assert len(versions) == 3

    def test_get_current_version_none(self):
        """Test current version is None when nothing deployed."""
        from model_rollback import ModelRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_dir=Path(tmpdir) / 'registry')
            assert registry.get_current_version() is None

    def test_register_with_deploy(self):
        """Test register + deploy in one call."""
        from model_rollback import ModelRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_dir=Path(tmpdir) / 'registry')

            model_path = Path(tmpdir) / 'model.keras'
            model_path.write_text('mock model')

            registry.register_model(
                model_path=model_path,
                version='1.0.0',
                metrics={'accuracy': 0.90},
                deploy=True,
            )

            assert registry.get_current_version() == '1.0.0'

    def test_deployment_history(self):
        """Test deployment history is recorded."""
        from model_rollback import ModelRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_dir=Path(tmpdir) / 'registry')

            for v in ['1.0.0', '2.0.0']:
                model_path = Path(tmpdir) / f'model_{v}.keras'
                model_path.write_text(f'mock {v}')
                registry.register_model(
                    model_path=model_path,
                    version=v,
                    metrics={'f1': 0.85},
                    deploy=True,
                )

            history = registry.get_deployment_history()
            assert len(history) >= 2
            assert history[-1]['action'] == 'deploy'


class TestRollbackValidator:
    """Tests for RollbackValidator class."""

    def test_validator_initialization(self):
        """Test validator creates successfully."""
        from model_rollback import RollbackValidator

        validator = RollbackValidator()
        assert validator is not None


class TestRollbackIntegration:
    """Integration tests for rollback workflow."""

    def test_full_rollback_workflow(self):
        """Test complete rollback workflow."""
        from model_rollback import ModelRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_dir=Path(tmpdir) / 'registry')

            # Register and deploy version 1
            model1 = Path(tmpdir) / 'model_v1.keras'
            model1.write_text('v1 data')
            registry.register_model(
                model_path=model1,
                version='1.0.0',
                metrics={'f1': 0.88},
                deploy=True,
            )

            # Register and deploy version 2 (worse)
            model2 = Path(tmpdir) / 'model_v2.keras'
            model2.write_text('v2 data')
            registry.register_model(
                model_path=model2,
                version='2.0.0',
                metrics={'f1': 0.82},
                deploy=True,
            )

            assert registry.get_current_version() == '2.0.0'

            # Rollback to v1
            success = registry.rollback(target_version='1.0.0')

            assert success is True
            assert registry.get_current_version() == '1.0.0'

    def test_rollback_to_previous(self):
        """Test rollback without specifying target (auto-finds previous)."""
        from model_rollback import ModelRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_dir=Path(tmpdir) / 'registry')

            for v in ['1.0.0', '2.0.0']:
                model_path = Path(tmpdir) / f'model_{v}.keras'
                model_path.write_text(f'{v} data')
                registry.register_model(
                    model_path=model_path,
                    version=v,
                    metrics={'f1': 0.85},
                    deploy=True,
                )

            assert registry.get_current_version() == '2.0.0'

            success = registry.rollback()

            assert success is True
            assert registry.get_current_version() == '1.0.0'

    def test_rollback_records_history(self):
        """Test that rollback is recorded in history."""
        from model_rollback import ModelRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_dir=Path(tmpdir) / 'registry')

            for v in ['1.0.0', '2.0.0']:
                model_path = Path(tmpdir) / f'model_{v}.keras'
                model_path.write_text(f'{v} data')
                registry.register_model(
                    model_path=model_path,
                    version=v,
                    metrics={'f1': 0.85},
                    deploy=True,
                )

            registry.rollback(target_version='1.0.0')

            history = registry.get_deployment_history()
            rollbacks = [h for h in history if h['action'] == 'rollback']
            assert len(rollbacks) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
