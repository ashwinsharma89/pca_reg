"""
Automated Scheduler for PCA-Agent
Schedule regular campaign analysis runs
"""

import schedule
import time
import logging
from datetime import datetime
from main import run_pca_agent
from db_connector import DatabaseConnector
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CampaignAnalysisScheduler:
    """
    Automated scheduler for running campaign analysis
    """
    
    def __init__(self, config: dict):
        """
        Initialize scheduler
        
        Args:
            config: Configuration dictionary with database, email settings
        """
        self.config = config
        self.db_connector = None
        
        if 'database' in config:
            self.db_connector = DatabaseConnector(config['database']['connection_string'])
    
    def run_analysis_job(self):
        """Execute the analysis job"""
        logger.info("=" * 80)
        logger.info(f"Starting scheduled analysis job at {datetime.now()}")
        logger.info("=" * 80)
        
        try:
            # Load data from database or file
            if self.db_connector:
                logger.info("Loading data from database...")
                df = self.db_connector.load_campaign_data(
                    table_name=self.config['database'].get('table_name'),
                    start_date=self.config.get('start_date'),
                    end_date=self.config.get('end_date')
                )
                
                # Save to temp file
                temp_file = 'temp_campaign_data.csv'
                df.to_csv(temp_file, index=False)
                data_path = temp_file
            else:
                data_path = self.config.get('data_path')
            
            # Run analysis
            logger.info("Running PCA-Agent analysis...")
            results = run_pca_agent(
                data_path=data_path,
                target_column=self.config.get('target_column', 'conversions'),
                use_sample_data=False if data_path else True,
                tune_hyperparameters=self.config.get('tune_hyperparameters', False)
            )
            
            # Save results to database if configured
            if self.db_connector and 'results_table' in self.config['database']:
                logger.info("Saving results to database...")
                self.db_connector.save_results(
                    results['results'],
                    self.config['database']['results_table']
                )
            
            # Send email notification if configured
            if 'email' in self.config:
                logger.info("Sending email notification...")
                self.send_email_report(results)
            
            # Clean up temp file
            if self.db_connector and os.path.exists(temp_file):
                os.unlink(temp_file)
            
            logger.info("✅ Analysis job completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Analysis job failed: {str(e)}")
            logger.exception(e)
            
            # Send error notification
            if 'email' in self.config:
                self.send_error_notification(str(e))
    
    def send_email_report(self, results: dict):
        """
        Send email with analysis results
        
        Args:
            results: Analysis results dictionary
        """
        try:
            email_config = self.config['email']
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = email_config['from']
            msg['To'] = ', '.join(email_config['to'])
            msg['Subject'] = f"PCA-Agent Analysis Report - {datetime.now().strftime('%Y-%m-%d')}"
            
            # Email body
            best_result = results['results'].iloc[0]
            body = f"""
            PCA-Agent Analysis Report
            ========================
            
            Analysis completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            Best Model: {results['model_name']}
            R² Score: {best_result['R2']:.4f}
            RMSE: {best_result['RMSE']:.2f}
            MAPE: {best_result['MAPE']:.2f}%
            
            Detailed results are attached.
            
            ---
            PCA-Agent Automated Reporting System
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach output files
            for filename in ['model_results.csv', 'feature_importance.csv', 'executive_summary.txt']:
                filepath = f'output/{filename}'
                if os.path.exists(filepath):
                    with open(filepath, 'rb') as f:
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(f.read())
                        encoders.encode_base64(part)
                        part.add_header('Content-Disposition', f'attachment; filename={filename}')
                        msg.attach(part)
            
            # Send email
            with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
                server.starttls()
                server.login(email_config['username'], email_config['password'])
                server.send_message(msg)
            
            logger.info("Email report sent successfully")
            
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
    
    def send_error_notification(self, error_message: str):
        """Send error notification email"""
        try:
            email_config = self.config['email']
            
            msg = MIMEMultipart()
            msg['From'] = email_config['from']
            msg['To'] = ', '.join(email_config['to'])
            msg['Subject'] = f"PCA-Agent Analysis FAILED - {datetime.now().strftime('%Y-%m-%d')}"
            
            body = f"""
            PCA-Agent Analysis Failed
            =========================
            
            The scheduled analysis job failed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            Error: {error_message}
            
            Please check the logs for more details.
            
            ---
            PCA-Agent Automated Reporting System
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
                server.starttls()
                server.login(email_config['username'], email_config['password'])
                server.send_message(msg)
            
            logger.info("Error notification sent")
            
        except Exception as e:
            logger.error(f"Failed to send error notification: {str(e)}")
    
    def start(self, schedule_time: str = "09:00"):
        """
        Start the scheduler
        
        Args:
            schedule_time: Time to run daily (HH:MM format)
        """
        logger.info(f"Starting scheduler - Analysis will run daily at {schedule_time}")
        
        # Schedule daily job
        schedule.every().day.at(schedule_time).do(self.run_analysis_job)
        
        # Also support weekly, monthly options
        if self.config.get('schedule_type') == 'weekly':
            day = self.config.get('schedule_day', 'monday')
            schedule.every().week.on(day).at(schedule_time).do(self.run_analysis_job)
            logger.info(f"Scheduled weekly on {day} at {schedule_time}")
        
        # Keep running
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute


# Example configuration
EXAMPLE_CONFIG = {
    'database': {
        'connection_string': 'postgresql://user:password@localhost:5432/campaign_db',
        'table_name': 'campaigns',
        'results_table': 'analysis_results',
        'start_date': None,  # None = all data
        'end_date': None
    },
    'email': {
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'username': 'your-email@gmail.com',
        'password': 'your-app-password',
        'from': 'pca-agent@company.com',
        'to': ['analyst@company.com', 'manager@company.com']
    },
    'target_column': 'conversions',
    'tune_hyperparameters': False,
    'schedule_type': 'daily',  # 'daily', 'weekly'
    'schedule_time': '09:00',
    'schedule_day': 'monday'  # for weekly
}


if __name__ == "__main__":
    # Load configuration (from file or environment)
    import json
    
    # Try to load from config file
    try:
        with open('scheduler_config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.warning("No config file found, using example config")
        config = EXAMPLE_CONFIG
    
    # Create and start scheduler
    scheduler = CampaignAnalysisScheduler(config)
    scheduler.start(config.get('schedule_time', '09:00'))
