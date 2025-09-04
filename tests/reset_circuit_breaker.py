#!/usr/bin/env python3
"""Reset circuit breaker and restart embedding processing"""

import asyncio
import logging
import sys
import os

# Add the services directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'services', 'orchestrator'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def reset_circuit_breaker():
    """Reset the circuit breaker states"""
    try:
        from embedding_load_balancer import get_load_balancer
        
        logger.info("🔄 Resetting circuit breakers...")
        
        # Get the load balancer instance
        load_balancer = get_load_balancer()
        
        # Initialize connections (this will discover and reset instances)
        await load_balancer.initialize_connections()
        
        # Reset all circuit breakers to closed state
        for port in load_balancer.instance_circuit_breakers:
            breaker = load_balancer.instance_circuit_breakers[port]
            breaker['state'] = 'closed'
            breaker['failures'] = 0
            breaker['last_failure_time'] = 0
            breaker['success_count'] = 0
            logger.info(f"✅ Reset circuit breaker for port {port}")
        
        # Test embedding generation
        logger.info("🧪 Testing embedding generation...")
        test_text = "This is a test for embedding generation"
        test_model = "granite-embedding"  # Use the model we know works
        
        embedding, success, error_msg = await load_balancer.generate_embedding(test_text, test_model)
        
        if success and embedding:
            logger.info(f"✅ Embedding test successful! Generated {len(embedding)}-dimensional embedding")
        else:
            logger.error(f"❌ Embedding test failed: {error_msg}")
            return False
            
        # Print stats
        stats = load_balancer.get_stats()
        logger.info(f"📊 Load balancer stats:")
        logger.info(f"  - Strategy: {stats['strategy']}")
        logger.info(f"  - Success rate: {stats['success_rate']:.2%}")
        logger.info(f"  - Active connections: {stats['active_connections']}")
        
        for port, instance_stats in stats['instance_stats'].items():
            logger.info(f"  - Port {port}: {instance_stats['circuit_breaker_state']} "
                       f"(failures: {instance_stats['circuit_breaker_failures']})")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to reset circuit breaker: {e}")
        return False

async def main():
    logger.info("🚀 Starting circuit breaker reset...")
    
    success = await reset_circuit_breaker()
    
    if success:
        logger.info("✅ Circuit breaker reset completed successfully!")
        logger.info("💡 The embedding system should now work properly.")
        logger.info("🔄 The ingestion process should continue automatically.")
    else:
        logger.error("❌ Circuit breaker reset failed.")
        logger.info("💡 Check the logs above for specific issues.")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)