#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Neo4j 连接诊断工具
帮助诊断和解决 Neo4j 连接问题
"""

import sys
import os
import time

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from config.config import load_env_file, config

def print_section(title):
    """打印章节标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def test_connection(password):
    """测试Neo4j连接"""
    try:
        from neo4j import GraphDatabase
        from neo4j.exceptions import AuthError, ServiceUnavailable
        
        driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, password)
        )
        driver.verify_connectivity()
        driver.close()
        return True, "连接成功"
    except AuthError as e:
        return False, f"认证失败: {e}"
    except ServiceUnavailable as e:
        return False, f"服务不可用: {e}"
    except Exception as e:
        return False, f"其他错误: {e}"

def main():
    print_section("Neo4j 连接诊断工具")
    
    # 1. 显示当前配置
    print_section("1. 当前配置")
    print(f"Neo4j URI:      {config.neo4j_uri}")
    print(f"Neo4j User:     {config.neo4j_user}")
    print(f"Neo4j Password: {'*' * len(config.neo4j_password)}")
    print(f"Neo4j Database: {config.neo4j_database}")
    
    # 2. 测试当前密码
    print_section("2. 测试当前配置")
    success, message = test_connection(config.neo4j_password)
    if success:
        print(f"✅ {message}")
        print("\n🎉 Neo4j 连接正常！您可以继续使用。")
        return 0
    else:
        print(f"❌ {message}")
    
    # 3. 尝试常见密码
    print_section("3. 尝试常见密码")
    common_passwords = ["neo4j", "password", "admin", "12345678"]
    
    for pwd in common_passwords:
        if pwd == config.neo4j_password:
            continue  # 已经测试过了
        
        print(f"\n尝试密码: '{pwd}'...")
        time.sleep(1)  # 避免速率限制
        success, message = test_connection(pwd)
        
        if success:
            print(f"✅ 成功！正确的密码是: '{pwd}'")
            print(f"\n💡 请更新 .env 文件中的 NEO4J_PASSWORD:")
            print(f"   NEO4J_PASSWORD={pwd}")
            
            # 询问是否自动更新
            try:
                response = input("\n是否自动更新 .env 文件? (y/n): ").strip().lower()
                if response == 'y':
                    update_env_file(pwd)
            except KeyboardInterrupt:
                print("\n操作已取消")
            return 0
        else:
            print(f"❌ 失败")
    
    # 4. 提供解决方案
    print_section("4. 解决方案")
    print("""
Neo4j 认证失败可能的原因和解决方案：

1. 密码不正确
   - 检查 .env 文件中的 NEO4J_PASSWORD
   - 确认 Neo4j 容器启动时设置的密码

2. Neo4j 速率限制
   - 多次认证失败后，Neo4j 会临时锁定
   - 解决方案：等待 5-10 分钟后重试
   - 或者重启 Neo4j 容器

3. Neo4j 服务未运行
   - 检查 Neo4j 是否正在运行：
     sudo docker ps | grep neo4j
   - 启动 Neo4j：
     sudo docker start neo4j
   - 或创建新容器：
     sudo docker run -d --name neo4j \\
       -p 7474:7474 -p 7687:7687 \\
       -e NEO4J_AUTH=neo4j/password \\
       -v neo4j_data:/data \\
       neo4j:latest

4. 重置 Neo4j（清除所有数据）
   sudo docker stop neo4j
   sudo docker rm neo4j
   sudo docker volume rm neo4j_data
   # 然后用新密码重新创建容器

💡 建议：
   - 访问 Neo4j Browser: http://localhost:7474
   - 使用浏览器界面验证连接和密码
    """)
    
    return 1

def update_env_file(new_password):
    """更新 .env 文件中的密码"""
    env_file = ".env"
    
    try:
        with open(env_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        updated = False
        for i, line in enumerate(lines):
            if line.strip().startswith('NEO4J_PASSWORD='):
                lines[i] = f'NEO4J_PASSWORD={new_password}\n'
                updated = True
                break
        
        if not updated:
            lines.append(f'\nNEO4J_PASSWORD={new_password}\n')
        
        with open(env_file, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print(f"✅ 已更新 {env_file}")
        print("   请重新运行您的脚本")
    except Exception as e:
        print(f"❌ 更新失败: {e}")

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n操作已取消")
        sys.exit(1)

