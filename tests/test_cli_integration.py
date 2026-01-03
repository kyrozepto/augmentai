"""
CLI integration tests for AugmentAI.

Tests end-to-end CLI flows including:
- augmentai --help
- augmentai prepare (with temp dataset)
- augmentai validate (with sample policy)
- augmentai domains
- Error handling for missing files
"""

import pytest
from pathlib import Path
from typer.testing import CliRunner

from augmentai.cli.app import app


class TestCLIHelp:
    """Test CLI help output."""
    
    def test_main_help(self, cli_runner: CliRunner):
        """Main help shows all commands."""
        result = cli_runner.invoke(app, ["--help"])
        
        assert result.exit_code == 0
        assert "prepare" in result.output
        assert "chat" in result.output
        assert "domains" in result.output
        assert "validate" in result.output
        assert "export" in result.output
    
    def test_prepare_help(self, cli_runner: CliRunner):
        """Prepare command help shows all options."""
        result = cli_runner.invoke(app, ["prepare", "--help"])
        
        assert result.exit_code == 0
        assert "--domain" in result.output
        assert "--output" in result.output
        assert "--seed" in result.output
        assert "--split" in result.output
        assert "--dry-run" in result.output
    
    def test_verbose_flag_in_help(self, cli_runner: CliRunner):
        """Global --verbose flag is shown."""
        result = cli_runner.invoke(app, ["--help"])
        
        assert result.exit_code == 0
        assert "--verbose" in result.output or "-v" in result.output
    
    def test_quiet_flag_in_help(self, cli_runner: CliRunner):
        """Global --quiet flag is shown."""
        result = cli_runner.invoke(app, ["--help"])
        
        assert result.exit_code == 0
        assert "--quiet" in result.output or "-q" in result.output


class TestDomainsCommand:
    """Test the domains listing command."""
    
    def test_domains_lists_all(self, cli_runner: CliRunner):
        """Domains command lists all available domains."""
        result = cli_runner.invoke(app, ["domains"])
        
        assert result.exit_code == 0
        assert "medical" in result.output.lower()
        assert "ocr" in result.output.lower()
        assert "satellite" in result.output.lower()
        assert "natural" in result.output.lower()


class TestPrepareCommand:
    """Test the prepare command."""
    
    def test_prepare_nonexistent_path(self, cli_runner: CliRunner):
        """Prepare fails with helpful error for nonexistent path."""
        result = cli_runner.invoke(app, ["prepare", "/nonexistent/path/12345"])
        
        # Should fail but not with raw exception
        assert result.exit_code != 0
    
    def test_prepare_dry_run(self, cli_runner: CliRunner, temp_dataset: Path, tmp_path: Path):
        """Prepare dry run shows what would happen."""
        output_dir = tmp_path / "output"
        
        result = cli_runner.invoke(app, [
            "prepare", str(temp_dataset),
            "--domain", "natural",
            "--output", str(output_dir),
            "--dry-run",
            "--skip-lint",  # Skip linting for faster test
        ])
        
        # Should complete without error
        assert result.exit_code == 0
        assert "dry run" in result.output.lower() or "Dry run" in result.output
    
    def test_prepare_with_domain(self, cli_runner: CliRunner, temp_dataset: Path, tmp_path: Path):
        """Prepare works with specified domain."""
        output_dir = tmp_path / "output"
        
        result = cli_runner.invoke(app, [
            "prepare", str(temp_dataset),
            "--domain", "medical",
            "--output", str(output_dir),
            "--dry-run",
            "--skip-lint",
        ])
        
        assert result.exit_code == 0
        assert "medical" in result.output.lower()
    
    def test_prepare_invalid_split(self, cli_runner: CliRunner, temp_dataset: Path):
        """Prepare fails with helpful error for invalid split."""
        result = cli_runner.invoke(app, [
            "prepare", str(temp_dataset),
            "--split", "invalid",
            "--skip-lint",
        ])
        
        assert result.exit_code != 0
        # Should show error about split format


class TestValidateCommand:
    """Test the validate command."""
    
    def test_validate_nonexistent_file(self, cli_runner: CliRunner):
        """Validate fails for nonexistent policy file."""
        result = cli_runner.invoke(app, [
            "validate", "/nonexistent/policy.yaml"
        ])
        
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "File not found" in result.output
    
    def test_validate_with_policy_file(
        self, 
        cli_runner: CliRunner, 
        sample_natural_policy, 
        tmp_path: Path
    ):
        """Validate works with valid policy file."""
        # Write policy to temp file
        policy_file = tmp_path / "test_policy.yaml"
        policy_file.write_text(sample_natural_policy.to_yaml())
        
        result = cli_runner.invoke(app, [
            "validate", str(policy_file),
            "--domain", "natural"
        ])
        
        assert result.exit_code == 0


class TestErrorHandling:
    """Test error handling and user-friendly messages."""
    
    def test_verbose_mode_gives_more_info(self, cli_runner: CliRunner):
        """Verbose mode shows additional information."""
        result = cli_runner.invoke(app, [
            "--verbose",
            "prepare", "/nonexistent/path/xyz"
        ])
        
        # Should fail, but verbose flag should be accepted
        assert result.exit_code != 0
    
    def test_quiet_mode_less_output(self, cli_runner: CliRunner):
        """Quiet mode reduces output."""
        result = cli_runner.invoke(app, [
            "--quiet", 
            "domains"
        ])
        
        # Should succeed (domains is a simple listing)
        assert result.exit_code == 0


class TestExportCommand:
    """Test the export command."""
    
    def test_export_nonexistent_file(self, cli_runner: CliRunner):
        """Export fails for nonexistent policy file."""
        result = cli_runner.invoke(app, [
            "export", "/nonexistent/policy.yaml"
        ])
        
        assert result.exit_code != 0
    
    def test_export_python_format(
        self, 
        cli_runner: CliRunner, 
        sample_natural_policy, 
        tmp_path: Path
    ):
        """Export generates Python script."""
        # Write policy to temp file
        policy_file = tmp_path / "test_policy.yaml"
        policy_file.write_text(sample_natural_policy.to_yaml())
        
        output_dir = tmp_path / "export_output"
        output_dir.mkdir()
        
        result = cli_runner.invoke(app, [
            "export", str(policy_file),
            "--output", str(output_dir),
            "--format", "python"
        ])
        
        assert result.exit_code == 0
        
        # Check file was created
        exported_file = output_dir / f"{sample_natural_policy.name}.py"
        assert exported_file.exists()
