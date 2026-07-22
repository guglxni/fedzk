#!/usr/bin/env python3
"""
Configuration Hot-Reloading System
=================================

Provides automatic configuration reloading when files or environment variables change.
Supports file watching, signal-based reloading, and callback mechanisms.
"""

import os
import time
import signal
import threading
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional
import logging
import hashlib


class ConfigHotReload:
    """Configuration hot-reloading manager."""

    def __init__(self, config_manager=None, watch_interval: int = 5):
        self.config_manager = config_manager
        self.watch_interval = watch_interval
        self.watch_threads: Dict[str, threading.Thread] = {}
        self.reload_callbacks: List[Callable] = []
        self.file_checksums: Dict[str, str] = {}
        self.environment_checksum: str = ""
        self.logger = logging.getLogger(__name__)
        self.running = False

        # Setup signal handlers for manual reload
        signal.signal(signal.SIGHUP, self._signal_reload_handler)

    def start_watching(self):
        """Start watching for configuration changes."""
        if self.running:
            self.logger.warning("Hot reload is already running")
            return

        self.running = True
        self.logger.info("Starting configuration hot-reload monitoring")

        # Watch configuration files
        self._watch_config_files()

        # Watch environment variables
        self._watch_environment()

        self.logger.info("Configuration hot-reload monitoring started")

    def stop_watching(self):
        """Stop watching for configuration changes."""
        if not self.running:
            return

        self.running = False
        self.logger.info("Stopping configuration hot-reload monitoring")

        # Stop all watch threads
        for thread_name, thread in self.watch_threads.items():
            if thread.is_alive():
                thread.join(timeout=1.0)

        self.watch_threads.clear()
        self.logger.info("Configuration hot-reload monitoring stopped")

    def add_reload_callback(self, callback: Callable):
        """Add a callback function to be called on configuration reload."""
        if callback not in self.reload_callbacks:
            self.reload_callbacks.append(callback)
            self.logger.debug(f"Added reload callback: {callback.__name__}")

    def remove_reload_callback(self, callback: Callable):
        """Remove a reload callback function."""
        if callback in self.reload_callbacks:
            self.reload_callbacks.remove(callback)
            self.logger.debug(f"Removed reload callback: {callback.__name__}")

    def force_reload(self):
        """Force a configuration reload."""
        self.logger.info("Forcing configuration reload")
        self._perform_reload("manual")

    def _watch_config_files(self):
        """Watch configuration files for changes."""
        config_files = [
            Path("./config/environment.yaml"),
            Path("./config/secrets.yaml"),
            Path("./config/database.yaml"),
            Path("./config/security.yaml")
        ]

        for config_file in config_files:
            if config_file.exists():
                thread = threading.Thread(
                    target=self._watch_file,
                    args=(config_file,),
                    name=f"watch-{config_file.name}",
                    daemon=True
                )
                thread.start()
                self.watch_threads[config_file.name] = thread

    def _watch_file(self, file_path: Path):
        """Watch a specific file for changes."""
        self.logger.debug(f"Starting to watch file: {file_path}")

        while self.running:
            try:
                if file_path.exists():
                    current_checksum = self._calculate_file_checksum(file_path)

                    if file_path.name not in self.file_checksums:
                        # First time seeing this file
                        self.file_checksums[file_path.name] = current_checksum
                    elif self.file_checksums[file_path.name] != current_checksum:
                        # File has changed
                        self.logger.info(f"Configuration file changed: {file_path}")
                        self.file_checksums[file_path.name] = current_checksum
                        self._perform_reload(f"file:{file_path.name}")

                time.sleep(self.watch_interval)

            except Exception as e:
                self.logger.error(f"Error watching file {file_path}: {e}")
                time.sleep(self.watch_interval)

    def _watch_environment(self):
        """Watch environment variables for changes."""
        thread = threading.Thread(
            target=self._watch_env_vars,
            name="watch-environment",
            daemon=True
        )
        thread.start()
        self.watch_threads["environment"] = thread

    def _watch_env_vars(self):
        """Watch environment variables for changes."""
        self.logger.debug("Starting to watch environment variables")

        while self.running:
            try:
                # Calculate checksum of relevant environment variables
                env_vars = {}
                fedzk_vars = [var for var in os.environ.keys() if var.startswith('FEDZK_')]

                for var in fedzk_vars:
                    env_vars[var] = os.environ[var]

                current_checksum = hashlib.md5(str(sorted(env_vars.items())).encode()).hexdigest()

                if not self.environment_checksum:
                    # First time
                    self.environment_checksum = current_checksum
                elif self.environment_checksum != current_checksum:
                    # Environment has changed
                    self.logger.info("Environment variables changed")
                    self.environment_checksum = current_checksum
                    self._perform_reload("environment")

                time.sleep(self.watch_interval)

            except Exception as e:
                self.logger.error(f"Error watching environment variables: {e}")
                time.sleep(self.watch_interval)

    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of a file."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating checksum for {file_path}: {e}")
            return ""

    def _perform_reload(self, source: str):
        """Perform the actual configuration reload."""
        try:
            self.logger.info(f"Reloading configuration (triggered by: {source})")

            # Reload configuration
            if self.config_manager:
                self.config_manager.load_from_environment()
                self.config_manager.load_from_file()
                self.config_manager.validate_configuration()

            # Notify callbacks
            for callback in self.reload_callbacks:
                try:
                    callback()
                except Exception as e:
                    self.logger.error(f"Reload callback failed: {e}")

            self.logger.info("Configuration reload completed successfully")

        except Exception as e:
            self.logger.error(f"Configuration reload failed: {e}")

    def _signal_reload_handler(self, signum, frame):
        """Handle SIGHUP signal for manual reload."""
        self.logger.info("Received SIGHUP signal, triggering configuration reload")
        self._perform_reload("signal")

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the hot reload system."""
        return {
            'running': self.running,
            'watch_interval': self.watch_interval,
            'active_threads': len([t for t in self.watch_threads.values() if t.is_alive()]),
            'watched_files': list(self.file_checksums.keys()),
            'reload_callbacks': len(self.reload_callbacks),
            'file_checksums': self.file_checksums.copy(),
            'environment_checksum': self.environment_checksum
        }


class ConfigChangeNotifier:
    """Configuration change notification system."""

    def __init__(self):
        self.listeners: Dict[str, List[Callable]] = {}
        self.logger = logging.getLogger(__name__)

    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to configuration change events."""
        if event_type not in self.listeners:
            self.listeners[event_type] = []

        if callback not in self.listeners[event_type]:
            self.listeners[event_type].append(callback)
            self.logger.debug(f"Subscribed to {event_type}: {callback.__name__}")

    def unsubscribe(self, event_type: str, callback: Callable):
        """Unsubscribe from configuration change events."""
        if event_type in self.listeners and callback in self.listeners[event_type]:
            self.listeners[event_type].remove(callback)
            self.logger.debug(f"Unsubscribed from {event_type}: {callback.__name__}")

    def notify(self, event_type: str, **kwargs):
        """Notify listeners of configuration changes."""
        if event_type in self.listeners:
            for callback in self.listeners[event_type]:
                try:
                    callback(**kwargs)
                except Exception as e:
                    self.logger.error(f"Notification callback failed: {e}")

    def get_subscriptions(self) -> Dict[str, int]:
        """Get subscription counts by event type."""
        return {event_type: len(listeners) for event_type, listeners in self.listeners.items()}


class ReloadableConfig:
    """Base class for reloadable configuration components."""

    def __init__(self):
        self.notifier = ConfigChangeNotifier()
        self.last_reload_time = time.time()

    def on_config_reload(self):
        """Called when configuration is reloaded."""
        self.last_reload_time = time.time()
        self.logger.info(f"{self.__class__.__name__} configuration reloaded")

        # Notify subscribers
        self.notifier.notify('config_reloaded', component=self.__class__.__name__)

    def subscribe_to_changes(self, callback: Callable):
        """Subscribe to configuration changes."""
        self.notifier.subscribe('config_reloaded', callback)

    def get_reload_info(self) -> Dict[str, Any]:
        """Get information about the last reload."""
        return {
            'last_reload_time': self.last_reload_time,
            'time_since_reload': time.time() - self.last_reload_time,
            'subscriptions': self.notifier.get_subscriptions()
        }


# Integration with the main configuration system
def setup_hot_reload(config_manager=None) -> ConfigHotReload:
    """Setup hot reload for the configuration system."""
    hot_reload = ConfigHotReload(config_manager)

    # Add reload callback to update the configuration
    def reload_callback():
        if config_manager:
            try:
                config_manager.load_from_environment()
                config_manager.load_from_file()
                config_manager.validate_configuration()
                print("✅ Configuration reloaded successfully")
            except Exception as e:
                print(f"❌ Configuration reload failed: {e}")

    hot_reload.add_reload_callback(reload_callback)

    return hot_reload


def demo_hot_reload():
    """Demonstrate hot reload functionality."""
    print("🔄 Configuration Hot-Reload Demonstration")
    print("=" * 50)

    # Create a functional config manager
    class FunctionalConfigManager:
        def __init__(self):
            self.value = "initial"
            self.last_modified = time.time()

        def load_from_environment(self):
            # Functional loading from environment
            new_value = os.getenv('DEMO_CONFIG', self.value)
            if new_value != self.value:
                print(f"Configuration updated from environment: {self.value} -> {new_value}")
                self.value = new_value
                self.last_modified = time.time()

        def load_from_file(self):
            # Functional loading from file
            config_file = Path('./demo_config.txt')
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        new_value = f.read().strip()
                    if new_value != self.value:
                        print(f"Configuration updated from file: {self.value} -> {new_value}")
                        self.value = new_value
                        self.last_modified = time.time()
                except Exception as e:
                    print(f"Error reading config file: {e}")

        def validate_configuration(self):
            # Functional validation
            if not self.value or len(self.value.strip()) == 0:
                print("Warning: Configuration value is empty")
                return False
            print(f"Configuration validated: {self.value}")
            return True

    # Setup hot reload
    config_manager = FunctionalConfigManager()
    hot_reload = setup_hot_reload(config_manager)

    print("1. Starting hot reload monitoring...")
    hot_reload.start_watching()

    print("2. Current configuration value:", config_manager.value)
    print("3. Try changing the configuration by:")
    print("   - Setting environment: export DEMO_CONFIG=new_value")
    print("   - Creating file: echo 'file_value' > demo_config.txt")
    print("   - Sending signal: kill -HUP <pid>")
    print("4. The configuration will automatically reload when changes are detected")

    try:
        # Keep running for demonstration
        time.sleep(30)
    except KeyboardInterrupt:
        print("\nStopping hot reload...")
    finally:
        hot_reload.stop_watching()
        print("Hot reload demonstration completed")


if __name__ == "__main__":
    demo_hot_reload()

