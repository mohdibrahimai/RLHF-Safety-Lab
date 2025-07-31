import json
import os
import streamlit as st
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum

class LogLevel(Enum):
    """Log level enumeration"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class Logger:
    """Logger for AI safety research platform activities"""
    
    def __init__(self, log_file: str = "data/activity.log", max_logs: int = 1000):
        self.log_file = log_file
        self.max_logs = max_logs
        self.logs = []
        self.ensure_log_dir()
        self.load_existing_logs()
    
    def ensure_log_dir(self):
        """Ensure log directory exists"""
        log_dir = os.path.dirname(self.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
    def load_existing_logs(self):
        """Load existing logs from file"""
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    self.logs = json.load(f)
                
                # Ensure logs don't exceed max limit
                if len(self.logs) > self.max_logs:
                    self.logs = self.logs[-self.max_logs:]
                    self.save_logs()
        except Exception as e:
            # If we can't load logs, start fresh
            self.logs = []
            self.log("warning", f"Could not load existing logs: {str(e)}")
    
    def save_logs(self):
        """Save logs to file"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.logs, f, indent=2)
        except Exception as e:
            # If we can't save logs, at least show error in Streamlit
            st.error(f"Failed to save logs: {str(e)}")
    
    def log(self, level: str, message: str, component: Optional[str] = None, 
            details: Optional[Dict[str, Any]] = None):
        """Log a message with specified level"""
        
        # Validate log level
        try:
            log_level = LogLevel(level.lower())
        except ValueError:
            log_level = LogLevel.INFO
        
        # Create log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": log_level.value,
            "message": message,
            "component": component,
            "details": details or {}
        }
        
        # Add to logs list
        self.logs.append(log_entry)
        
        # Maintain max logs limit
        if len(self.logs) > self.max_logs:
            self.logs = self.logs[-self.max_logs:]
        
        # Save to file
        self.save_logs()
        
        # Also display in Streamlit if appropriate
        self._display_log_in_streamlit(log_entry)
    
    def _display_log_in_streamlit(self, log_entry: Dict[str, Any]):
        """Display log in Streamlit interface based on level"""
        level = log_entry["level"]
        message = log_entry["message"]
        
        # Only display warning and error messages as toast notifications
        # Info and debug messages are just stored for later viewing
        if level == LogLevel.ERROR.value or level == LogLevel.CRITICAL.value:
            st.error(f"🔴 {message}")
        elif level == LogLevel.WARNING.value:
            st.warning(f"⚠️ {message}")
    
    def debug(self, message: str, component: Optional[str] = None, 
              details: Optional[Dict[str, Any]] = None):
        """Log debug message"""
        self.log("debug", message, component, details)
    
    def info(self, message: str, component: Optional[str] = None, 
             details: Optional[Dict[str, Any]] = None):
        """Log info message"""
        self.log("info", message, component, details)
    
    def warning(self, message: str, component: Optional[str] = None, 
                details: Optional[Dict[str, Any]] = None):
        """Log warning message"""
        self.log("warning", message, component, details)
    
    def error(self, message: str, component: Optional[str] = None, 
              details: Optional[Dict[str, Any]] = None):
        """Log error message"""
        self.log("error", message, component, details)
    
    def critical(self, message: str, component: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        """Log critical message"""
        self.log("critical", message, component, details)
    
    def get_recent_logs(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent log entries"""
        return self.logs[-count:] if self.logs else []
    
    def get_logs_by_level(self, level: str) -> List[Dict[str, Any]]:
        """Get logs filtered by level"""
        try:
            target_level = LogLevel(level.lower())
            return [log for log in self.logs if log.get("level") == target_level.value]
        except ValueError:
            return []
    
    def get_logs_by_component(self, component: str) -> List[Dict[str, Any]]:
        """Get logs filtered by component"""
        return [log for log in self.logs if log.get("component") == component]
    
    def get_logs_by_timeframe(self, start_time: datetime, end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get logs within a specific timeframe"""
        if end_time is None:
            end_time = datetime.now()
        
        filtered_logs = []
        for log in self.logs:
            try:
                log_time = datetime.fromisoformat(log["timestamp"])
                if start_time <= log_time <= end_time:
                    filtered_logs.append(log)
            except (ValueError, KeyError):
                continue
        
        return filtered_logs
    
    def search_logs(self, search_term: str) -> List[Dict[str, Any]]:
        """Search logs by message content"""
        search_term = search_term.lower()
        return [
            log for log in self.logs 
            if search_term in log.get("message", "").lower()
        ]
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get statistics about logged activities"""
        if not self.logs:
            return {"total": 0, "by_level": {}, "by_component": {}}
        
        stats = {
            "total": len(self.logs),
            "by_level": {},
            "by_component": {},
            "timerange": {
                "earliest": self.logs[0]["timestamp"] if self.logs else None,
                "latest": self.logs[-1]["timestamp"] if self.logs else None
            }
        }
        
        # Count by level
        for log in self.logs:
            level = log.get("level", "unknown")
            stats["by_level"][level] = stats["by_level"].get(level, 0) + 1
        
        # Count by component
        for log in self.logs:
            component = log.get("component", "unknown")
            stats["by_component"][component] = stats["by_component"].get(component, 0) + 1
        
        return stats
    
    def clear_logs(self):
        """Clear all logs"""
        self.logs = []
        self.save_logs()
        self.info("Logs cleared", "logger")
    
    def export_logs(self, format: str = "json") -> str:
        """Export logs in specified format"""
        if format.lower() == "json":
            return json.dumps(self.logs, indent=2)
        elif format.lower() == "csv":
            return self._logs_to_csv()
        elif format.lower() == "txt":
            return self._logs_to_text()
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _logs_to_csv(self) -> str:
        """Convert logs to CSV format"""
        if not self.logs:
            return "timestamp,level,component,message,details\n"
        
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(["timestamp", "level", "component", "message", "details"])
        
        # Write data
        for log in self.logs:
            writer.writerow([
                log.get("timestamp", ""),
                log.get("level", ""),
                log.get("component", ""),
                log.get("message", ""),
                json.dumps(log.get("details", {}))
            ])
        
        return output.getvalue()
    
    def _logs_to_text(self) -> str:
        """Convert logs to plain text format"""
        if not self.logs:
            return "No logs available\n"
        
        lines = []
        for log in self.logs:
            timestamp = log.get("timestamp", "")
            level = log.get("level", "").upper()
            component = log.get("component", "")
            message = log.get("message", "")
            
            line = f"[{timestamp}] {level}"
            if component:
                line += f" ({component})"
            line += f": {message}"
            
            lines.append(line)
        
        return "\n".join(lines)
    
    def log_model_operation(self, operation: str, model_name: str, 
                           status: str, details: Optional[Dict] = None):
        """Log model-related operations"""
        message = f"Model operation: {operation} on {model_name} - {status}"
        self.info(message, "model_loader", details)
    
    def log_training_progress(self, epoch: int, loss: float, 
                             additional_metrics: Optional[Dict] = None):
        """Log training progress"""
        message = f"Training epoch {epoch} completed with loss: {loss:.4f}"
        details = {"epoch": epoch, "loss": loss}
        if additional_metrics:
            details.update(additional_metrics)
        self.info(message, "preference_trainer", details)
    
    def log_evaluation_result(self, metric_name: str, value: float, 
                             comparison: Optional[str] = None):
        """Log evaluation results"""
        message = f"Evaluation {metric_name}: {value:.4f}"
        if comparison:
            message += f" ({comparison})"
        details = {"metric": metric_name, "value": value}
        self.info(message, "harm_evaluator", details)
    
    def log_data_operation(self, operation: str, data_type: str, 
                          count: int, success: bool = True):
        """Log data operations"""
        status = "successful" if success else "failed"
        message = f"Data operation: {operation} {count} {data_type} - {status}"
        level = "info" if success else "error"
        self.log(level, message, "data_handler", {"operation": operation, "data_type": data_type, "count": count})
    
    def log_user_action(self, action: str, page: str, 
                       details: Optional[Dict] = None):
        """Log user actions"""
        message = f"User action: {action} on {page}"
        self.info(message, "user_interface", details)
    
    def get_formatted_logs_for_display(self, count: int = 20) -> List[str]:
        """Get formatted logs for display in Streamlit"""
        recent_logs = self.get_recent_logs(count)
        formatted_logs = []
        
        for log in recent_logs:
            timestamp = log.get("timestamp", "")
            level = log.get("level", "").upper()
            component = log.get("component", "")
            message = log.get("message", "")
            
            # Format timestamp for display
            try:
                dt = datetime.fromisoformat(timestamp)
                formatted_time = dt.strftime("%H:%M:%S")
            except:
                formatted_time = timestamp
            
            # Create formatted log line
            formatted_log = f"[{formatted_time}] {level}"
            if component:
                formatted_log += f" ({component})"
            formatted_log += f": {message}"
            
            formatted_logs.append(formatted_log)
        
        return formatted_logs
