"""
FEDzk Log Aggregation and Analysis System
==========================================

Advanced log aggregation with:
- ELK stack integration (Elasticsearch, Logstash, Kibana)
- Real-time log analysis and alerting
- Performance metrics and anomaly detection
- Log correlation and pattern recognition
"""

import json
import logging
import time
import threading
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque
import statistics

try:
    from elasticsearch import Elasticsearch, helpers
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class LogAggregator:
    """
    Advanced log aggregation and analysis system
    """

    def __init__(self, retention_hours: int = 24, max_buffer_size: int = 10000):
        self.retention_hours = retention_hours
        self.max_buffer_size = max_buffer_size

        # Data structures for aggregation
        self.logs_by_level = defaultdict(list)
        self.logs_by_service = defaultdict(list)
        self.logs_by_time = deque(maxlen=max_buffer_size)
        self.error_patterns = defaultdict(int)
        self.performance_metrics = defaultdict(list)

        # Anomaly detection
        self.baseline_metrics = {}
        self.anomaly_thresholds = {
            'error_rate': 0.1,  # 10% error rate threshold
            'response_time': 5.0,  # 5 second response time threshold
            'throughput_drop': 0.5  # 50% throughput drop threshold
        }

        # Threading
        self.lock = threading.Lock()
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()

    def aggregate_log(self, log_entry: Dict[str, Any]):
        """Aggregate a log entry for analysis"""
        with self.lock:
            timestamp = log_entry.get('timestamp_epoch', time.time())

            # Add to time-based storage
            self.logs_by_time.append((timestamp, log_entry))

            # Aggregate by level
            level = log_entry.get('level', 'UNKNOWN')
            self.logs_by_level[level].append((timestamp, log_entry))

            # Aggregate by service
            service = log_entry.get('service', 'unknown')
            self.logs_by_service[service].append((timestamp, log_entry))

            # Extract error patterns
            if level in ['ERROR', 'CRITICAL']:
                self._extract_error_patterns(log_entry)

            # Extract performance metrics
            if log_entry.get('performance_metric'):
                self._extract_performance_metrics(log_entry)

    def _extract_error_patterns(self, log_entry: Dict[str, Any]):
        """Extract common error patterns for analysis"""
        message = log_entry.get('message', '').lower()

        patterns = {
            'connection_error': r'connection.*(?:failed|refused|timeout)',
            'authentication_error': r'auth.*(?:failed|invalid|unauthorized)',
            'validation_error': r'validation.*error|invalid.*input',
            'timeout_error': r'timeout|timed.*out',
            'memory_error': r'memory.*error|out.*of.*memory',
            'zk_error': r'zk.*error|zero.*knowledge.*error',
            'crypto_error': r'crypto.*error|encryption.*error|decryption.*error'
        }

        for pattern_name, pattern in patterns.items():
            if re.search(pattern, message):
                self.error_patterns[pattern_name] += 1

    def _extract_performance_metrics(self, log_entry: Dict[str, Any]):
        """Extract performance metrics for analysis"""
        operation = log_entry.get('operation', 'unknown')
        duration = log_entry.get('duration_seconds', 0)
        success = log_entry.get('success', True)

        if operation and duration > 0:
            self.performance_metrics[operation].append({
                'duration': duration,
                'success': success,
                'timestamp': log_entry.get('timestamp_epoch', time.time())
            })

    def get_statistics(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get comprehensive log statistics"""
        with self.lock:
            cutoff_time = time.time() - (time_window_minutes * 60)

            stats = {
                'time_window_minutes': time_window_minutes,
                'total_logs': len(self.logs_by_time),
                'logs_by_level': {},
                'logs_by_service': {},
                'error_patterns': dict(self.error_patterns),
                'performance_insights': self._get_performance_insights(cutoff_time),
                'anomalies': self._detect_anomalies(cutoff_time)
            }

            # Count logs by level in time window
            for level, logs in self.logs_by_level.items():
                recent_logs = [log for ts, log in logs if ts > cutoff_time]
                stats['logs_by_level'][level] = len(recent_logs)

            # Count logs by service in time window
            for service, logs in self.logs_by_service.items():
                recent_logs = [log for ts, log in logs if ts > cutoff_time]
                stats['logs_by_service'][service] = len(recent_logs)

            return stats

    def _get_performance_insights(self, cutoff_time: float) -> Dict[str, Any]:
        """Generate performance insights from aggregated data"""
        insights = {}

        for operation, metrics in self.performance_metrics.items():
            recent_metrics = [
                m for m in metrics
                if m['timestamp'] > cutoff_time
            ]

            if recent_metrics:
                durations = [m['duration'] for m in recent_metrics]
                success_rate = sum(1 for m in recent_metrics if m['success']) / len(recent_metrics)

                insights[operation] = {
                    'count': len(recent_metrics),
                    'avg_duration': statistics.mean(durations),
                    'median_duration': statistics.median(durations),
                    'p95_duration': statistics.quantiles(durations, n=20)[18] if len(durations) >= 20 else max(durations),
                    'success_rate': success_rate,
                    'slow_operations': len([d for d in durations if d > 5.0])
                }

        return insights

    def _detect_anomalies(self, cutoff_time: float) -> List[Dict[str, Any]]:
        """Detect anomalies in log patterns"""
        anomalies = []

        # Check error rate anomalies
        total_logs = len([ts for ts, _ in self.logs_by_time if ts > cutoff_time])
        error_logs = len([ts for ts, log in self.logs_by_level.get('ERROR', []) if ts > cutoff_time])

        if total_logs > 0:
            error_rate = error_logs / total_logs
            if error_rate > self.anomaly_thresholds['error_rate']:
                anomalies.append({
                    'type': 'high_error_rate',
                    'severity': 'high',
                    'message': f'Error rate {error_rate:.2%} exceeds threshold {self.anomaly_thresholds["error_rate"]:.2%}',
                    'value': error_rate,
                    'threshold': self.anomaly_thresholds['error_rate']
                })

        # Check performance anomalies
        for operation, insights in self._get_performance_insights(cutoff_time).items():
            if insights['p95_duration'] > self.anomaly_thresholds['response_time']:
                anomalies.append({
                    'type': 'slow_response',
                    'severity': 'medium',
                    'operation': operation,
                    'message': f'P95 response time {insights["p95_duration"]:.2f}s exceeds threshold',
                    'value': insights['p95_duration'],
                    'threshold': self.anomaly_thresholds['response_time']
                })

        return anomalies

    def _cleanup_worker(self):
        """Background worker to clean up old log entries"""
        while True:
            try:
                time.sleep(300)  # Clean up every 5 minutes
                self._cleanup_old_entries()
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")

    def _cleanup_old_entries(self):
        """Remove log entries older than retention period"""
        with self.lock:
            cutoff_time = time.time() - (self.retention_hours * 3600)

            # Clean up time-based storage
            while self.logs_by_time and self.logs_by_time[0][0] < cutoff_time:
                self.logs_by_time.popleft()

            # Clean up level-based storage
            for level in self.logs_by_level:
                self.logs_by_level[level] = [
                    (ts, log) for ts, log in self.logs_by_level[level]
                    if ts > cutoff_time
                ]

            # Clean up service-based storage
            for service in self.logs_by_service:
                self.logs_by_service[service] = [
                    (ts, log) for ts, log in self.logs_by_service[service]
                    if ts > cutoff_time
                ]

            # Clean up performance metrics (keep last 1000 per operation)
            for operation in self.performance_metrics:
                self.performance_metrics[operation] = self.performance_metrics[operation][-1000:]


class ElasticsearchLogShipper:
    """
    Ship logs to Elasticsearch for advanced analysis
    """

    def __init__(self, hosts: List[str], index_prefix: str = "fedzk-logs",
                 batch_size: int = 100, flush_interval: int = 30):
        if not ELASTICSEARCH_AVAILABLE:
            raise ImportError("Elasticsearch client not available")

        self.es_client = Elasticsearch(hosts=hosts)
        self.index_prefix = index_prefix
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        self.buffer = []
        self.last_flush = time.time()
        self.lock = threading.Lock()

        # Start background flusher
        self.flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
        self.flush_thread.start()

    def ship_log(self, log_entry: Dict[str, Any]):
        """Add log entry to shipping buffer"""
        with self.lock:
            self.buffer.append(log_entry)

            # Flush if buffer is full
            if len(self.buffer) >= self.batch_size:
                self._flush_buffer()

    def _flush_worker(self):
        """Background worker to periodically flush logs"""
        while True:
            try:
                time.sleep(self.flush_interval)
                with self.lock:
                    if self.buffer:
                        self._flush_buffer()
            except Exception as e:
                logger.error(f"Error in flush worker: {e}")

    def _flush_buffer(self):
        """Flush buffered logs to Elasticsearch"""
        if not self.buffer:
            return

        try:
            actions = []
            for log_entry in self.buffer:
                # Create index name with date
                timestamp = log_entry.get('timestamp', datetime.utcnow().isoformat())
                date = timestamp.split('T')[0].replace('-', '.')

                index_name = f"{self.index_prefix}-{date}"

                actions.extend([
                    {"index": {"_index": index_name, "_id": log_entry.get('event_id')}},
                    log_entry
                ])

            if actions:
                success, failed = helpers.bulk(self.es_client, actions, raise_on_error=False)

                if failed:
                    logger.error(f"Failed to index {len(failed)} log entries")
                else:
                    logger.debug(f"Successfully indexed {len(self.buffer)} log entries")

            self.buffer.clear()
            self.last_flush = time.time()

        except Exception as e:
            logger.error(f"Error flushing logs to Elasticsearch: {e}")


class RedisLogBuffer:
    """
    Redis-based log buffering for high-throughput scenarios
    """

    def __init__(self, host: str = 'localhost', port: int = 6379,
                 key_prefix: str = "fedzk:logs", max_buffer_size: int = 1000):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis client not available")

        self.redis_client = redis.Redis(host=host, port=port)
        self.key_prefix = key_prefix
        self.max_buffer_size = max_buffer_size

    def buffer_log(self, log_entry: Dict[str, Any], priority: str = "normal"):
        """Buffer log entry in Redis"""
        key = f"{self.key_prefix}:{priority}"
        log_json = json.dumps(log_entry)

        # Add to list with timestamp for ordering
        timestamp = time.time()
        self.redis_client.zadd(key, {log_json: timestamp})

        # Trim buffer to max size
        self.redis_client.zremrangebyrank(key, 0, -self.max_buffer_size - 1)

    def get_buffered_logs(self, priority: str = "normal", count: int = 100) -> List[Dict[str, Any]]:
        """Retrieve buffered logs from Redis"""
        key = f"{self.key_prefix}:{priority}"
        log_entries = self.redis_client.zrange(key, 0, count - 1, withscores=True)

        logs = []
        for log_json, score in log_entries:
            try:
                log_entry = json.loads(log_json)
                log_entry['_buffer_timestamp'] = score
                logs.append(log_entry)
            except json.JSONDecodeError:
                continue

        return logs

    def clear_buffer(self, priority: str = "normal"):
        """Clear log buffer"""
        key = f"{self.key_prefix}:{priority}"
        self.redis_client.delete(key)


class LogAnalyzer:
    """
    Advanced log analysis and pattern recognition
    """

    def __init__(self):
        self.patterns = {
            'error_sequences': [],
            'performance_trends': [],
            'security_incidents': [],
            'resource_usage': []
        }

    def analyze_error_sequence(self, logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze sequences of errors for patterns"""
        error_sequences = []
        current_sequence = []

        for log in logs:
            if log.get('level') in ['ERROR', 'CRITICAL']:
                current_sequence.append(log)
            else:
                if len(current_sequence) >= 3:  # Significant error sequence
                    error_sequences.append({
                        'start_time': current_sequence[0]['timestamp'],
                        'end_time': current_sequence[-1]['timestamp'],
                        'error_count': len(current_sequence),
                        'error_types': list(set(log.get('message', '') for log in current_sequence)),
                        'affected_services': list(set(log.get('service', '') for log in current_sequence))
                    })
                current_sequence = []

        return error_sequences

    def analyze_performance_trends(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        performance_logs = [log for log in logs if log.get('performance_metric')]

        if not performance_logs:
            return {}

        # Group by operation
        operations = defaultdict(list)
        for log in performance_logs:
            operation = log.get('operation', 'unknown')
            operations[operation].append(log.get('duration_seconds', 0))

        trends = {}
        for operation, durations in operations.items():
            if len(durations) >= 5:  # Need minimum data points
                sorted_durations = sorted(durations)
                trends[operation] = {
                    'avg_duration': statistics.mean(durations),
                    'trend': 'increasing' if durations[-1] > statistics.mean(durations[:-1]) else 'decreasing',
                    'volatility': statistics.stdev(durations) if len(durations) > 1 else 0,
                    'outliers': len([d for d in durations if d > sorted_durations[int(len(sorted_durations) * 0.95)]])
                }

        return trends

    def detect_security_incidents(self, logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect potential security incidents"""
        security_logs = [log for log in logs if log.get('security_event') or log.get('level') == 'WARNING']

        incidents = []
        failed_auth_count = 0
        suspicious_activity_count = 0

        for log in security_logs:
            message = log.get('message', '').lower()

            if 'authentication failed' in message or 'invalid credentials' in message:
                failed_auth_count += 1
            elif 'suspicious' in message or 'unusual' in message:
                suspicious_activity_count += 1

        if failed_auth_count >= 5:
            incidents.append({
                'type': 'brute_force_attempt',
                'severity': 'high',
                'description': f'{failed_auth_count} failed authentication attempts detected',
                'recommendation': 'Implement account lockout or rate limiting'
            })

        if suspicious_activity_count >= 3:
            incidents.append({
                'type': 'suspicious_activity',
                'severity': 'medium',
                'description': f'{suspicious_activity_count} suspicious activities detected',
                'recommendation': 'Review access patterns and user behavior'
            })

        return incidents


class FEDzkLogAnalysisPipeline:
    """
    Complete log analysis pipeline with all components
    """

    def __init__(self,
                 elasticsearch_hosts: Optional[List[str]] = None,
                 redis_host: str = 'localhost',
                 enable_analysis: bool = True):
        self.aggregator = LogAggregator()
        self.analyzer = LogAnalyzer() if enable_analysis else None

        # Optional components
        self.es_shipper = None
        if elasticsearch_hosts and ELASTICSEARCH_AVAILABLE:
            self.es_shipper = ElasticsearchLogShipper(elasticsearch_hosts)

        self.redis_buffer = None
        if REDIS_AVAILABLE:
            self.redis_buffer = RedisLogBuffer(redis_host)

    def process_log(self, log_entry: Dict[str, Any]):
        """Process a log entry through the entire pipeline"""
        # Aggregate for real-time analysis
        self.aggregator.aggregate_log(log_entry)

        # Ship to Elasticsearch if configured
        if self.es_shipper:
            self.es_shipper.ship_log(log_entry)

        # Buffer in Redis if configured
        if self.redis_buffer:
            priority = 'high' if log_entry.get('level') in ['ERROR', 'CRITICAL'] else 'normal'
            self.redis_buffer.buffer_log(log_entry, priority)

    def get_analysis_report(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'time_window_minutes': time_window_minutes,
            'statistics': self.aggregator.get_statistics(time_window_minutes)
        }

        if self.analyzer:
            # Get recent logs for analysis
            recent_logs = []
            cutoff_time = time.time() - (time_window_minutes * 60)

            # Collect recent logs from aggregator
            for level_logs in self.aggregator.logs_by_level.values():
                recent_logs.extend([
                    log for ts, log in level_logs
                    if ts > cutoff_time
                ])

            report.update({
                'error_sequences': self.analyzer.analyze_error_sequence(recent_logs),
                'performance_trends': self.analyzer.analyze_performance_trends(recent_logs),
                'security_incidents': self.analyzer.detect_security_incidents(recent_logs)
            })

        return report

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the log analysis pipeline"""
        status = {
            'aggregator': {
                'healthy': True,
                'buffered_logs': len(self.aggregator.logs_by_time),
                'error_patterns': len(self.aggregator.error_patterns)
            }
        }

        if self.es_shipper:
            status['elasticsearch'] = {
                'healthy': True,
                'buffered_logs': len(self.es_shipper.buffer),
                'last_flush': self.es_shipper.last_flush
            }

        if self.redis_buffer:
            status['redis'] = {
                'healthy': True,
                'buffer_size': self.redis_buffer.redis_client.llen(f"{self.redis_buffer.key_prefix}:normal")
            }

        return status


# Global pipeline instance
_pipeline = None


def get_log_pipeline() -> FEDzkLogAnalysisPipeline:
    """Get or create global log analysis pipeline"""
    global _pipeline
    if _pipeline is None:
        _pipeline = FEDzkLogAnalysisPipeline()
    return _pipeline


def initialize_log_aggregation(elasticsearch_hosts: Optional[List[str]] = None,
                              redis_host: str = 'localhost',
                              enable_analysis: bool = True) -> FEDzkLogAnalysisPipeline:
    """
    Initialize the complete log aggregation and analysis system
    """
    global _pipeline

    _pipeline = FEDzkLogAnalysisPipeline(
        elasticsearch_hosts=elasticsearch_hosts,
        redis_host=redis_host,
        enable_analysis=enable_analysis
    )

    logger.info("Initialized FEDzk log aggregation and analysis system",
               extra={"structured_data": {"log_aggregation_initialized": True}})

    return _pipeline


# Integration helpers
def create_elk_pipeline(es_hosts: List[str]) -> FEDzkLogAnalysisPipeline:
    """Create ELK stack integrated pipeline"""
    return initialize_log_aggregation(
        elasticsearch_hosts=es_hosts,
        enable_analysis=True
    )


def create_redis_buffered_pipeline(redis_host: str = 'localhost') -> FEDzkLogAnalysisPipeline:
    """Create Redis-buffered pipeline for high throughput"""
    return initialize_log_aggregation(
        elasticsearch_hosts=None,
        redis_host=redis_host,
        enable_analysis=True
    )


def get_log_insights(time_window_minutes: int = 60) -> Dict[str, Any]:
    """Get comprehensive log insights"""
    pipeline = get_log_pipeline()
    return pipeline.get_analysis_report(time_window_minutes)
