#!/usr/bin/env python3
"""
System validation script to check if all components are working
"""

import asyncio
import subprocess
import time
import httpx
import yaml
from pathlib import Path

async def check_service_health(service_name, url, timeout=5):
    """Check if a service is healthy"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=timeout)
            if response.status_code == 200:
                print(f"✅ {service_name} is healthy ({url})")
                return True
            else:
                print(f"⚠️  {service_name} returned status {response.status_code}")
                return False
    except Exception as e:
        print(f"❌ {service_name} is not accessible: {e}")
        return False

async def validate_system():
    """Validate the entire system"""
    print("🔍 SYSTEM VALIDATION")
    print("=" * 50)
    
    # Check configuration
    config_file = Path(__file__).parent / "shared" / "configs" / "system.yaml"
    if not config_file.exists():
        print("❌ System configuration file not found")
        return False
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    print("✅ Configuration file loaded successfully")
    
    # Check if services are running
    services = {
        "Orchestrator": f"http://{config['services']['orchestrator']['host']}:{config['services']['orchestrator']['port']}/health",
        "Model Gateway": f"http://{config['services']['model_gateway']['host']}:{config['services']['model_gateway']['port']}/health",
        "Vector Engine": f"http://{config['services']['vector_engine']['host']}:{config['services']['vector_engine']['port']}/",
    }
    
    print("\n🌐 CHECKING SERVICES")
    print("-" * 30)
    
    health_results = []
    for service_name, url in services.items():
        result = await check_service_health(service_name, url)
        health_results.append(result)
    
    # Check databases
    print("\n🗄️  CHECKING DATABASES")
    print("-" * 30)
    
    # Check PostgreSQL
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=config['databases']['postgresql']['host'],
            port=config['databases']['postgresql']['port'],
            database=config['databases']['postgresql']['database'],
            user=config['databases']['postgresql']['username'],
            password=config['databases']['postgresql']['password'],
            connect_timeout=5
        )
        conn.close()
        print("✅ PostgreSQL is accessible")
        pg_healthy = True
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        pg_healthy = False
    
    # Check Qdrant
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(
            host=config['databases']['qdrant']['host'],
            port=config['databases']['qdrant']['port'],
            timeout=5
        )
        collections = client.get_collections()
        print("✅ Qdrant is accessible")
        qdrant_healthy = True
    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
        qdrant_healthy = False
    
    # Check Ollama
    print("\n🤖 CHECKING OLLAMA")
    print("-" * 30)
    
    ollama_healthy = await check_service_health(
        "Ollama", 
        config['services']['model_gateway']['ollama_url'] + "/api/tags"
    )
    
    # Summary
    print("\n📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    total_checks = len(services) + 3  # services + postgres + qdrant + ollama
    passed_checks = sum(health_results) + pg_healthy + qdrant_healthy + ollama_healthy
    
    print(f"Services Health: {sum(health_results)}/{len(services)} ✅")
    print(f"Database Health: {pg_healthy + qdrant_healthy}/2 ✅")
    print(f"Ollama Health: {ollama_healthy}/1 ✅")
    print(f"Overall: {passed_checks}/{total_checks} checks passed")
    
    if passed_checks == total_checks:
        print("\n🎉 System is fully operational!")
        return True
    elif passed_checks >= total_checks * 0.7:  # 70% threshold
        print("\n⚠️  System is partially operational")
        return True
    else:
        print("\n💥 System has significant issues")
        return False

def check_build_status():
    """Check if all services can be built"""
    print("\n🔨 CHECKING BUILD STATUS")
    print("-" * 30)
    
    # Check Zig build
    zig_dir = Path(__file__).parent / "services" / "vector-engine"
    if zig_dir.exists():
        try:
            result = subprocess.run(
                ["zig", "build"],
                cwd=zig_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                print("✅ Zig vector engine builds successfully")
            else:
                print(f"❌ Zig build failed: {result.stderr}")
        except FileNotFoundError:
            print("⚠️  Zig compiler not found")
        except Exception as e:
            print(f"❌ Zig build error: {e}")
    
    # Check Python dependencies
    services = ["orchestrator", "model-gateway"]
    for service in services:
        req_file = Path(__file__).parent / "services" / service / "requirements.txt"
        if req_file.exists():
            print(f"✅ {service} requirements.txt found")
        else:
            print(f"⚠️  {service} requirements.txt missing")

async def main():
    """Main validation function"""
    print("🚀 AgenticRAG System Validation")
    print("=" * 50)
    
    # Check build status first
    check_build_status()
    
    # Then validate runtime
    system_healthy = await validate_system()
    
    print("\n" + "=" * 50)
    if system_healthy:
        print("✅ VALIDATION COMPLETE - System Ready")
        return 0
    else:
        print("❌ VALIDATION FAILED - System Needs Attention")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)