
import json
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import hashlib
import pickle
from pathlib import Path

from arvs.core.data_types import (
    SystemTelemetry, RobotState, RiskAssessment, MVISequence,
    Action, SystemMode, SafetyConstraints
)
from arvs.core.constants import (
    TELEMETRY_LOG_PATH, DECISION_LOG_PATH, SAFETY_LOG_PATH,
    SYSTEM_VERSION, IMPLEMENTATION_VERSION
)

logger = logging.getLogger(__name__)

class LogLevel(Enum):
    """Log levels for audit system."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class AuditEventType(Enum):
    """Types of audit events."""
    SYSTEM_START = "system_start"
    SYSTEM_SHUTDOWN = "system_shutdown"
    MODE_CHANGE = "mode_change"
    RISK_ASSESSMENT = "risk_assessment"
    DECISION_MADE = "decision_made"
    ACTION_EXECUTED = "action_executed"
    SAFETY_VIOLATION = "safety_violation"
    FAULT_DETECTED = "fault_detected"
    OPTIMIZATION = "optimization"
    LEARNING_UPDATE = "learning_update"
    TELEMETRY = "telemetry"

@dataclass
class AuditEvent:
    """An audit event for logging."""
    event_id: str
    event_type: AuditEventType
    timestamp: float
    robot_id: str
    system_mode: SystemMode
    log_level: LogLevel
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Generate event ID if not provided."""
        if not self.event_id:
            self.event_id = self._generate_event_id()
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        base = f"{self.event_type.value}_{self.timestamp}_{self.robot_id}"
        return hashlib.md5(base.encode()).hexdigest()[:8]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['event_type'] = self.event_type.value
        result['system_mode'] = self.system_mode.name
        result['log_level'] = self.log_level.value
        result['timestamp_human'] = datetime.fromtimestamp(self.timestamp).isoformat()
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str, indent=2)

class AuditLogger:
    """
    Centralized audit logging system for ARVS.
    
    Provides comprehensive logging for:
    1. System state changes
    2. Decision making process
    3. Safety violations
    4. Performance metrics
    5. Human-readable explanations
    
    Implements thread-safe logging with configurable backends.
    """
    
    def __init__(self, robot_id: str, log_directory: str = "/tmp/arvs_logs"):
        """
        Initialize audit logger.
        
        Args:
            robot_id: Robot identifier
            log_directory: Directory for log files
        """
        self.robot_id = robot_id
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # Log file paths
        self.telemetry_log_path = self.log_directory / "telemetry.log"
        self.decision_log_path = self.log_directory / "decisions.log"
        self.safety_log_path = self.log_directory / "safety.log"
        self.audit_log_path = self.log_directory / "audit.log"
        
        # Event storage
        self.event_buffer: List[AuditEvent] = []
        self.max_buffer_size = 1000
        self.buffer_lock = threading.Lock()
        
        # File handles
        self.telemetry_file = None
        self.decision_file = None
        self.safety_file = None
        self.audit_file = None
        
        # Statistics
        self.stats = {
            'total_events': 0,
            'events_by_type': {et.value: 0 for et in AuditEventType},
            'events_by_level': {lvl.value: 0 for lvl in LogLevel},
            'last_flush_time': time.time(),
            'disk_usage': 0
        }
        
        # Configuration
        self.enabled = True
        self.flush_interval = 5.0  # Flush to disk every 5 seconds
        self.max_log_size = 100 * 1024 * 1024  # 100MB max per log file
        self.compression_enabled = False
        
        # Start flush thread
        self.flush_thread = threading.Thread(target=self._flush_thread_func, daemon=True)
        self.flush_thread.start()
        
        # Initial system start event
        self.log_system_start()
        
        logger.info(f"Audit logger initialized for robot {robot_id}, logs at {log_directory}")
    
    def log_system_start(self):
        """Log system startup event."""
        event = AuditEvent(
            event_id="",
            event_type=AuditEventType.SYSTEM_START,
            timestamp=time.time(),
            robot_id=self.robot_id,
            system_mode=SystemMode.NORMAL,
            log_level=LogLevel.INFO,
            data={
                'system_version': SYSTEM_VERSION,
                'implementation_version': IMPLEMENTATION_VERSION,
                'log_directory': str(self.log_directory)
            },
            metadata={
                'pid': __import__('os').getpid(),
                'hostname': __import__('socket').gethostname()
            }
        )
        self._log_event(event)
    
    def log_system_shutdown(self):
        """Log system shutdown event."""
        event = AuditEvent(
            event_id="",
            event_type=AuditEventType.SYSTEM_SHUTDOWN,
            timestamp=time.time(),
            robot_id=self.robot_id,
            system_mode=SystemMode.NORMAL,
            log_level=LogLevel.INFO,
            data={'reason': 'normal_shutdown'},
            metadata={}
        )
        self._log_event(event)
        self.flush_all()  # Ensure all logs are written
    
    def log_mode_change(self, old_mode: SystemMode, new_mode: SystemMode,
                       reason: str, additional_data: Dict = None):
        """Log system mode change."""
        event = AuditEvent(
            event_id="",
            event_type=AuditEventType.MODE_CHANGE,
            timestamp=time.time(),
            robot_id=self.robot_id,
            system_mode=new_mode,
            log_level=LogLevel.WARNING if new_mode != SystemMode.NORMAL else LogLevel.INFO,
            data={
                'old_mode': old_mode.name,
                'new_mode': new_mode.name,
                'reason': reason,
                **(additional_data or {})
            },
            metadata={}
        )
        self._log_event(event)
    
    def log_risk_assessment(self, risk_assessment: RiskAssessment,
                           system_mode: SystemMode):
        """Log risk assessment event."""
        event = AuditEvent(
            event_id="",
            event_type=AuditEventType.RISK_ASSESSMENT,
            timestamp=time.time(),
            robot_id=self.robot_id,
            system_mode=system_mode,
            log_level=self._risk_to_log_level(risk_assessment.overall_risk),
            data={
                'overall_risk': risk_assessment.overall_risk,
                'component_risks': risk_assessment.component_risks,
                'risk_factors': risk_assessment.risk_factors,
                'confidence': risk_assessment.confidence
            },
            metadata={
                'timestamp': risk_assessment.timestamp
            }
        )
        self._log_event(event)
    
    def log_decision(self, mvi_sequence: MVISequence, system_mode: SystemMode,
                    optimization_metrics: Dict, safety_check_passed: bool):
        """Log decision making event."""
        event = AuditEvent(
            event_id="",
            event_type=AuditEventType.DECISION_MADE,
            timestamp=time.time(),
            robot_id=self.robot_id,
            system_mode=system_mode,
            log_level=LogLevel.INFO,
            data={
                'sequence_id': mvi_sequence.sequence_id,
                'action_count': len(mvi_sequence.actions),
                'expected_duration': mvi_sequence.expected_duration,
                'predicted_risk': mvi_sequence.predicted_risk.overall_risk,
                'optimization_metrics': optimization_metrics,
                'safety_check_passed': safety_check_passed,
                'actions': [
                    {
                        'action_id': action.action_id,
                        'action_type': action.action_type,
                        'duration': action.duration,
                        'priority': action.priority
                    }
                    for action in mvi_sequence.actions
                ]
            },
            metadata={
                'decision_timestamp': mvi_sequence.predicted_risk.timestamp
            }
        )
        self._log_event(event)
    
    def log_action_execution(self, action: Action, result: Dict,
                            system_mode: SystemMode):
        """Log action execution event."""
        event = AuditEvent(
            event_id="",
            event_type=AuditEventType.ACTION_EXECUTED,
            timestamp=time.time(),
            robot_id=self.robot_id,
            system_mode=system_mode,
            log_level=LogLevel.INFO if result.get('success', False) else LogLevel.ERROR,
            data={
                'action_id': action.action_id,
                'action_type': action.action_type,
                'result': result,
                'duration': action.duration
            },
            metadata={}
        )
        self._log_event(event)
    
    def log_safety_violation(self, violation_type: str, component: str,
                            value: float, limit: float, system_mode: SystemMode,
                            action_id: str = None):
        """Log safety violation event."""
        event = AuditEvent(
            event_id="",
            event_type=AuditEventType.SAFETY_VIOLATION,
            timestamp=time.time(),
            robot_id=self.robot_id,
            system_mode=system_mode,
            log_level=LogLevel.ERROR if value > limit * 1.5 else LogLevel.WARNING,
            data={
                'violation_type': violation_type,
                'component': component,
                'value': value,
                'limit': limit,
                'excess_ratio': value / limit if limit > 0 else float('inf'),
                'action_id': action_id
            },
            metadata={}
        )
        self._log_event(event)
    
    def log_fault_detected(self, fault_type: str, severity: str,
                          component: str, system_mode: SystemMode):
        """Log fault detection event."""
        event = AuditEvent(
            event_id="",
            event_type=AuditEventType.FAULT_DETECTED,
            timestamp=time.time(),
            robot_id=self.robot_id,
            system_mode=system_mode,
            log_level=self._fault_severity_to_log_level(severity),
            data={
                'fault_type': fault_type,
                'severity': severity,
                'component': component
            },
            metadata={}
        )
        self._log_event(event)
    
    def log_optimization(self, problem_id: str, num_variables: int,
                        solver_type: str, solve_time: float,
                        success: bool, system_mode: SystemMode):
        """Log optimization event."""
        event = AuditEvent(
            event_id="",
            event_type=AuditEventType.OPTIMIZATION,
            timestamp=time.time(),
            robot_id=self.robot_id,
            system_mode=system_mode,
            log_level=LogLevel.INFO if success else LogLevel.ERROR,
            data={
                'problem_id': problem_id,
                'num_variables': num_variables,
                'solver_type': solver_type,
                'solve_time': solve_time,
                'success': success
            },
            metadata={}
        )
        self._log_event(event)
    
    def log_learning_update(self, model_type: str, improvement: float,
                           samples_used: int, success: bool,
                           safety_violations: List[str], system_mode: SystemMode):
        """Log learning update event."""
        event = AuditEvent(
            event_id="",
            event_type=AuditEventType.LEARNING_UPDATE,
            timestamp=time.time(),
            robot_id=self.robot_id,
            system_mode=system_mode,
            log_level=LogLevel.INFO if success else LogLevel.WARNING,
            data={
                'model_type': model_type,
                'improvement': improvement,
                'samples_used': samples_used,
                'success': success,
                'safety_violations': safety_violations
            },
            metadata={}
        )
        self._log_event(event)
    
    def log_telemetry(self, telemetry: SystemTelemetry):
        """Log system telemetry event."""
        event = AuditEvent(
            event_id="",
            event_type=AuditEventType.TELEMETRY,
            timestamp=time.time(),
            robot_id=self.robot_id,
            system_mode=telemetry.system_mode,
            log_level=LogLevel.INFO,
            data=telemetry.to_dict(),
            metadata={}
        )
        self._log_event(event)
    
    def _risk_to_log_level(self, risk: float) -> LogLevel:
        """Convert risk value to appropriate log level."""
        if risk > 0.8:
            return LogLevel.ERROR
        elif risk > 0.6:
            return LogLevel.WARNING
        else:
            return LogLevel.INFO
    
    def _fault_severity_to_log_level(self, severity: str) -> LogLevel:
        """Convert fault severity to log level."""
        severity_map = {
            'CRITICAL': LogLevel.CRITICAL,
            'MAJOR': LogLevel.ERROR,
            'MODERATE': LogLevel.WARNING,
            'MINOR': LogLevel.INFO
        }
        return severity_map.get(severity.upper(), LogLevel.INFO)
    
    def _log_event(self, event: AuditEvent):
        """Internal method to log an event."""
        if not self.enabled:
            return
        
        with self.buffer_lock:
            self.event_buffer.append(event)
            self.stats['total_events'] += 1
            self.stats['events_by_type'][event.event_type.value] += 1
            self.stats['events_by_level'][event.log_level.value] += 1
            
            # Flush if buffer is full
            if len(self.event_buffer) >= self.max_buffer_size:
                self._flush_buffer()
    
    def _flush_buffer(self):
        """Flush event buffer to appropriate log files."""
        if not self.event_buffer:
            return
        
        with self.buffer_lock:
            events_to_flush = self.event_buffer.copy()
            self.event_buffer.clear()
        
        # Open log files if not already open
        self._ensure_files_open()
        
        try:
            for event in events_to_flush:
                event_dict = event.to_dict()
                event_json = json.dumps(event_dict, default=str) + '\n'
                
                # Write to appropriate log file based on event type
                if event.event_type == AuditEventType.TELEMETRY:
                    if self.telemetry_file:
                        self.telemetry_file.write(event_json)
                
                elif event.event_type == AuditEventType.DECISION_MADE:
                    if self.decision_file:
                        self.decision_file.write(event_json)
                
                elif (event.event_type == AuditEventType.SAFETY_VIOLATION or 
                      event.event_type == AuditEventType.FAULT_DETECTED):
                    if self.safety_file:
                        self.safety_file.write(event_json)
                
                # Always write to audit log
                if self.audit_file:
                    self.audit_file.write(event_json)
            
            # Update disk usage
            self._update_disk_usage()
            
        except Exception as e:
            logger.error(f"Error flushing audit buffer: {e}")
            # Re-add events to buffer
            with self.buffer_lock:
                self.event_buffer.extend(events_to_flush)
    
    def _ensure_files_open(self):
        """Ensure log files are open."""
        try:
            if not self.telemetry_file:
                self.telemetry_file = open(self.telemetry_log_path, 'a')
            if not self.decision_file:
                self.decision_file = open(self.decision_log_path, 'a')
            if not self.safety_file:
                self.safety_file = open(self.safety_log_path, 'a')
            if not self.audit_file:
                self.audit_file = open(self.audit_log_path, 'a')
        except Exception as e:
            logger.error(f"Error opening log files: {e}")
    
    def _update_disk_usage(self):
        """Update disk usage statistics."""
        try:
            total_size = 0
            for log_file in [self.telemetry_log_path, self.decision_log_path,
                           self.safety_log_path, self.audit_log_path]:
                if log_file.exists():
                    total_size += log_file.stat().st_size
            
            self.stats['disk_usage'] = total_size
            
            # Rotate logs if they get too large
            if total_size > self.max_log_size:
                self._rotate_logs()
                
        except Exception as e:
            logger.error(f"Error updating disk usage: {e}")
    
    def _rotate_logs(self):
        """Rotate log files when they get too large."""
        logger.info("Rotating log files")
        
        # Close files
        self._close_files()
        
        # Create backup with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for log_path in [self.telemetry_log_path, self.decision_log_path,
                        self.safety_log_path, self.audit_log_path]:
            if log_path.exists():
                backup_path = log_path.parent / f"{log_path.stem}_{timestamp}{log_path.suffix}"
                try:
                    log_path.rename(backup_path)
                    logger.info(f"Rotated {log_path} to {backup_path}")
                except Exception as e:
                    logger.error(f"Error rotating {log_path}: {e}")
        
        # Reopen files
        self._ensure_files_open()
    
    def _close_files(self):
        """Close all log files."""
        for file_handle in [self.telemetry_file, self.decision_file,
                          self.safety_file, self.audit_file]:
            if file_handle:
                try:
                    file_handle.close()
                except Exception:
                    pass
        
        self.telemetry_file = None
        self.decision_file = None
        self.safety_file = None
        self.audit_file = None
    
    def _flush_thread_func(self):
        """Background thread for periodic flushing."""
        while True:
            time.sleep(self.flush_interval)
            self._flush_buffer()
            self.stats['last_flush_time'] = time.time()
    
    def flush_all(self):
        """Flush all buffered events immediately."""
        self._flush_buffer()
        if self.telemetry_file:
            self.telemetry_file.flush()
        if self.decision_file:
            self.decision_file.flush()
        if self.safety_file:
            self.safety_file.flush()
        if self.audit_file:
            self.audit_file.flush()
    
    def get_events(self, event_type: Optional[AuditEventType] = None,
                  log_level: Optional[LogLevel] = None,
                  start_time: Optional[float] = None,
                  end_time: Optional[float] = None,
                  limit: int = 100) -> List[AuditEvent]:
        """
        Retrieve events matching criteria.
        
        Args:
            event_type: Filter by event type
            log_level: Filter by log level
            start_time: Earliest timestamp
            end_time: Latest timestamp
            limit: Maximum number of events to return
            
        Returns:
            List of matching AuditEvents
        """
        # Read from audit log file
        events = []
        
        if not self.audit_log_path.exists():
            return events
        
        try:
            with open(self.audit_log_path, 'r') as f:
                lines = f.readlines()
            
            # Parse events
            for line in reversed(lines[-limit*2:]):  # Read extra to account for filtering
                try:
                    event_dict = json.loads(line.strip())
                    
                    # Convert back to AuditEvent
                    event = AuditEvent(
                        event_id=event_dict['event_id'],
                        event_type=AuditEventType(event_dict['event_type']),
                        timestamp=event_dict['timestamp'],
                        robot_id=event_dict['robot_id'],
                        system_mode=SystemMode[event_dict['system_mode']],
                        log_level=LogLevel(event_dict['log_level']),
                        data=event_dict['data'],
                        metadata=event_dict['metadata']
                    )
                    
                    # Apply filters
                    if event_type and event.event_type != event_type:
                        continue
                    if log_level and event.log_level != log_level:
                        continue
                    if start_time and event.timestamp < start_time:
                        continue
                    if end_time and event.timestamp > end_time:
                        continue
                    
                    events.append(event)
                    
                    if len(events) >= limit:
                        break
                        
                except (json.JSONDecodeError, KeyError):
                    continue
                    
        except Exception as e:
            logger.error(f"Error reading audit log: {e}")
        
        return events
    
    def generate_report(self, start_time: float, end_time: float) -> Dict[str, Any]:
        """
        Generate audit report for time period.
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            Report dictionary
        """
        events = self.get_events(start_time=start_time, end_time=end_time, limit=1000)
        
        # Count by type and level
        type_counts = {}
        level_counts = {}
        
        for event in events:
            type_counts[event.event_type.value] = type_counts.get(event.event_type.value, 0) + 1
            level_counts[event.log_level.value] = level_counts.get(event.log_level.value, 0) + 1
        
        # Find critical events
        critical_events = []
        for event in events:
            if (event.log_level in [LogLevel.ERROR, LogLevel.CRITICAL] or
                event.event_type in [AuditEventType.SAFETY_VIOLATION, 
                                   AuditEventType.FAULT_DETECTED]):
                critical_events.append({
                    'timestamp': event.timestamp,
                    'type': event.event_type.value,
                    'level': event.log_level.value,
                    'data': event.data
                })
        
        # Calculate statistics
        total_events = len(events)
        if total_events > 0:
            error_rate = level_counts.get('ERROR', 0) / total_events
            warning_rate = level_counts.get('WARNING', 0) / total_events
        else:
            error_rate = warning_rate = 0.0
        
        return {
            'period': {
                'start': start_time,
                'end': end_time,
                'duration': end_time - start_time
            },
            'summary': {
                'total_events': total_events,
                'error_rate': error_rate,
                'warning_rate': warning_rate,
                'event_type_distribution': type_counts,
                'log_level_distribution': level_counts
            },
            'critical_events': critical_events[:10],  # Top 10 critical events
            'system_stats': self.stats.copy()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audit logger statistics."""
        return {
            **self.stats,
            'enabled': self.enabled,
            'buffer_size': len(self.event_buffer),
            'log_directory': str(self.log_directory)
        }
    
    def enable(self, enabled: bool = True):
        """Enable or disable audit logging."""
        self.enabled = enabled
        if not enabled:
            self.flush_all()
        logger.info(f"Audit logging {'enabled' if enabled else 'disabled'}")
    
    def shutdown(self):
        """Shutdown audit logger."""
        self.enabled = False
        self.flush_all()
        self._close_files()
        logger.info("Audit logger shutdown")