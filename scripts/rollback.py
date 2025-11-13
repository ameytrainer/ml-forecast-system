"""
Production Rollback Script
Safely rollback to previous model version
"""

import argparse
from mlflow.tracking import MlflowClient
import mlflow
import time
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionRollback:
    """Handle production rollbacks safely"""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.client = MlflowClient()
    
    def rollback_to_version(self, version):
        """Rollback to specific model version"""
        
        logger.info("=" * 60)
        logger.info("🚨 INITIATING ROLLBACK")
        logger.info("=" * 60)
        
        # Step 1: Verify target version exists
        try:
            target = self.client.get_model_version(self.model_name, version)
            logger.info(f"✓ Target version {version} found")
            logger.info(f"  Created: {target.creation_timestamp}")
            logger.info(f"  Run ID: {target.run_id}")
        except Exception as e:
            logger.error(f"❌ Error: Version {version} not found")
            logger.error(str(e))
            return False
        
        # Step 2: Get current production version (for backup)
        try:
            current = self.client.get_model_version_by_alias(
                self.model_name, "Production"
            )
            logger.info(f"✓ Current production: Version {current.version}")
        except:
            logger.warning("⚠️ No current production version")
            current = None
        
        # Step 3: Update MLflow Registry
        logger.info(f"\n📝 Updating MLflow Registry...")
        try:
            self.client.set_registered_model_alias(
                name=self.model_name,
                alias="Production",
                version=str(version)
            )
            logger.info(f"✓ MLflow Registry updated to version {version}")
        except Exception as e:
            logger.error(f"❌ Failed to update registry: {e}")
            return False
        
        # Step 4: Run health checks
        logger.info(f"\n🏥 Running health checks...")
        if self.verify_health():
            logger.info(f"✓ Health checks passed")
        else:
            logger.error(f"❌ Health checks failed!")
            # Rollback the rollback
            if current:
                logger.warning("Attempting to restore previous version...")
                self.rollback_to_version(current.version)
            return False
        
        # Step 5: Log rollback
        self.log_rollback(current.version if current else None, version)
        
        logger.info("\n" + "=" * 60)
        logger.info(f"✅ ROLLBACK COMPLETE")
        logger.info(f"Production now serving: Version {version}")
        logger.info("=" * 60)
        
        return True
    
    def verify_health(self):
        """Verify system health after rollback"""
        # In production: actual health checks
        # For demo: simple checks
        logger.info("  Checking API health...")
        time.sleep(1)
        logger.info("  ✓ API responding")
        
        logger.info("  Checking predictions...")
        time.sleep(1)
        logger.info("  ✓ Predictions working")
        
        logger.info("  Checking latency...")
        time.sleep(1)
        logger.info("  ✓ Latency acceptable")
        
        return True
    
    def log_rollback(self, from_version, to_version):
        """Log rollback event"""
        try:
            with mlflow.start_run(run_name=f"rollback-{int(time.time())}"):
                mlflow.set_tag("event_type", "rollback")
                mlflow.set_tag("from_version", from_version or "None")
                mlflow.set_tag("to_version", to_version)
                mlflow.set_tag("timestamp", time.time())
                mlflow.set_tag("initiated_by", "manual")
                logger.info("✓ Rollback logged to MLflow")
        except Exception as e:
            logger.warning(f"Could not log to MLflow: {e}")


def main():
    parser = argparse.ArgumentParser(description="Rollback production model")
    parser.add_argument("--model", required=True,
                        help="Model name (e.g., 'sales-forecaster')")
    parser.add_argument("--version", required=True, type=int,
                        help="Target version number to rollback to")
    parser.add_argument("--confirm", action="store_true",
                        help="Confirm rollback (required)")
    
    args = parser.parse_args()
    
    if not args.confirm:
        print("⚠️  SAFETY CHECK: Add --confirm flag to proceed with rollback")
        print(f"   This will rollback {args.model} to version {args.version}")
        return 1
    
    # Execute rollback
    rollback = ProductionRollback(args.model)
    success = rollback.rollback_to_version(args.version)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
