"""
Neo4j图数据库管理器
用于存储和查询光电知识图谱的实体和关系
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime
import time

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable, AuthError, ClientError
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("⚠️ Neo4j driver not installed. Please install: pip install neo4j")

from config.config import config
from .knowledge_graph import Entity, Relation, EntityType, RelationType

class Neo4jManager:
    """Neo4j数据库管理器"""
    
    def __init__(self, max_retries: int = 3, retry_delay: int = 2):
        """初始化Neo4j连接
        
        Args:
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
        """
        if not NEO4J_AVAILABLE:
            raise ImportError("Neo4j driver not available")
            
        self.driver = None
        self.database = config.neo4j_database
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # 尝试连接并进行重试
        last_error = None
        for attempt in range(max_retries):
            try:
                self.driver = GraphDatabase.driver(
                    config.neo4j_uri,
                    auth=(config.neo4j_user, config.neo4j_password)
                )
                # 验证连接
                self.driver.verify_connectivity()
                print(f"✓ Successfully connected to Neo4j: {config.neo4j_uri}")
                self._create_constraints()
                return
            except AuthError as e:
                last_error = e
                print(f"✗ Authentication failed (attempt {attempt + 1}/{max_retries})")
                print(f"  Neo4j URI: {config.neo4j_uri}")
                print(f"  Neo4j User: {config.neo4j_user}")
                print(f"  Error: {e}")
                if attempt < max_retries - 1:
                    # 认证失败时等待更长时间以避免速率限制
                    wait_time = retry_delay * (attempt + 1) * 2
                    print(f"  Waiting {wait_time} seconds before retry to avoid rate limiting...")
                    time.sleep(wait_time)
            except ServiceUnavailable as e:
                last_error = e
                print(f"✗ Neo4j service unavailable (attempt {attempt + 1}/{max_retries})")
                print(f"  Error: {e}")
                if attempt < max_retries - 1:
                    print(f"  Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
            except Exception as e:
                last_error = e
                print(f"✗ Failed to connect to Neo4j (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"  Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
        
        # 所有重试都失败了
        print("\n" + "=" * 70)
        print("❌ Failed to connect to Neo4j after multiple attempts")
        print("=" * 70)
        print("\n🔧 Troubleshooting steps:")
        print("  1. Ensure Neo4j service is running:")
        print("     docker-compose up -d neo4j")
        print("  2. Check Neo4j configuration in .env file:")
        print(f"     NEO4J_URI={config.neo4j_uri}")
        print(f"     NEO4J_USER={config.neo4j_user}")
        print(f"     NEO4J_PASSWORD=***")
        print(f"     NEO4J_DATABASE={config.neo4j_database}")
        print("  3. If authentication failed, wait a few minutes before retrying")
        print("     (Neo4j has rate limiting for failed authentication attempts)")
        print("  4. Verify credentials by accessing Neo4j Browser:")
        print("     http://localhost:7474")
        print("=" * 70 + "\n")
        raise last_error
    
    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()
    
    def _create_constraints(self):
        """创建数据库约束和索引"""
        constraints = [
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT relation_id_unique IF NOT EXISTS FOR (r:Relation) REQUIRE r.id IS UNIQUE",
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX relation_type_index IF NOT EXISTS FOR (r:Relation) ON (r.type)"
        ]
        
        with self.driver.session(database=self.database) as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    # 约束可能已存在，忽略错误
                    pass
        
        print("✓ Neo4j constraints and indexes created")
    
    def insert_entity(self, entity: Entity) -> bool:
        """
        插入或更新实体
        - 基于实体ID进行MERGE（ID已经是基于名称+类型生成的唯一标识）
        - 合并多个描述、来源论文和属性
        """
        cypher = """
        MERGE (e:Entity {id: $id})
        ON CREATE SET
            e.name = $name,
            e.type = $type,
            e.description = $description,
            e.descriptions = $descriptions,
            e.properties = $properties,
            e.source_papers = $source_papers,
            e.confidence = $confidence,
            e.created_at = $created_at,
            e.updated_at = datetime()
        ON MATCH SET
            e.name = $name,
            e.type = $type,
            e.description = $description,
            e.descriptions = CASE 
                WHEN e.descriptions IS NULL THEN $descriptions
                ELSE [d IN e.descriptions WHERE NOT d IN $descriptions] + $descriptions
            END,
            e.source_papers = CASE 
                WHEN e.source_papers IS NULL THEN $source_papers
                ELSE [p IN e.source_papers WHERE NOT p IN $source_papers] + $source_papers
            END,
            e.confidence = CASE 
                WHEN $confidence > e.confidence THEN $confidence 
                ELSE e.confidence 
            END,
            e.properties = $properties,
            e.updated_at = datetime()
        RETURN e
        """
        
        try:
            # 准备descriptions列表
            descriptions = entity.descriptions if entity.descriptions else []
            if entity.description and entity.description not in descriptions:
                descriptions.append(entity.description)
            
            with self.driver.session(database=self.database) as session:
                result = session.run(cypher, {
                    'id': entity.id,
                    'name': entity.name,
                    'type': entity.type.value,
                    'description': entity.description,
                    'descriptions': descriptions,
                    'properties': json.dumps(entity.properties),
                    'source_papers': entity.source_papers,
                    'confidence': entity.confidence,
                    'created_at': entity.created_at
                })
                return result.single() is not None
        except Exception as e:
            print(f"插入实体失败: {e}")
            return False
    
    def insert_relation(self, relation: Relation) -> bool:
        """插入关系"""
        cypher = """
        MATCH (source:Entity {id: $source_id})
        MATCH (target:Entity {id: $target_id})
        MERGE (source)-[r:RELATES {id: $id, type: $type}]->(target)
        SET r.description = $description,
            r.properties = $properties,
            r.source_papers = $source_papers,
            r.confidence = $confidence,
            r.created_at = $created_at,
            r.updated_at = datetime()
        RETURN r
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(cypher, {
                    'id': relation.id,
                    'source_id': relation.source_id,
                    'target_id': relation.target_id,
                    'type': relation.type.value,
                    'description': relation.description,
                    'properties': json.dumps(relation.properties),
                    'source_papers': relation.source_papers,
                    'confidence': relation.confidence,
                    'created_at': relation.created_at
                })
                return result.single() is not None
        except Exception as e:
            print(f"插入关系失败: {e}")
            return False
    
    def batch_insert_entities(self, entities: List[Entity]) -> int:
        """
        批量插入或更新实体
        - 基于实体ID进行MERGE
        - 自动合并多个描述、来源论文和属性
        """
        success_count = 0
        
        with self.driver.session(database=self.database) as session:
            with session.begin_transaction() as tx:
                for entity in entities:
                    cypher = """
                    MERGE (e:Entity {id: $id})
                    ON CREATE SET
                        e.name = $name,
                        e.type = $type,
                        e.description = $description,
                        e.descriptions = $descriptions,
                        e.properties = $properties,
                        e.source_papers = $source_papers,
                        e.confidence = $confidence,
                        e.created_at = $created_at,
                        e.updated_at = datetime()
                    ON MATCH SET
                        e.name = $name,
                        e.type = $type,
                        e.description = $description,
                        e.descriptions = CASE 
                            WHEN e.descriptions IS NULL THEN $descriptions
                            ELSE [d IN e.descriptions WHERE NOT d IN $descriptions] + $descriptions
                        END,
                        e.source_papers = CASE 
                            WHEN e.source_papers IS NULL THEN $source_papers
                            ELSE [p IN e.source_papers WHERE NOT p IN $source_papers] + $source_papers
                        END,
                        e.confidence = CASE 
                            WHEN $confidence > e.confidence THEN $confidence 
                            ELSE e.confidence 
                        END,
                        e.properties = $properties,
                        e.updated_at = datetime()
                    """
                    
                    try:
                        # 准备descriptions列表
                        descriptions = entity.descriptions if entity.descriptions else []
                        if entity.description and entity.description not in descriptions:
                            descriptions.append(entity.description)
                        
                        tx.run(cypher, {
                            'id': entity.id,
                            'name': entity.name,
                            'type': entity.type.value,
                            'description': entity.description,
                            'descriptions': descriptions,
                            'properties': json.dumps(entity.properties),
                            'source_papers': entity.source_papers,
                            'confidence': entity.confidence,
                            'created_at': entity.created_at
                        })
                        success_count += 1
                    except Exception as e:
                        print(f"插入实体 {entity.id} 失败: {e}")
        
        return success_count
    
    def add_entities_batch(self, entities: List[Entity]) -> int:
        """批量添加实体（别名方法，保持兼容性）"""
        return self.batch_insert_entities(entities)
    
    def batch_insert_relations(self, relations: List[Relation]) -> int:
        """批量插入关系"""
        success_count = 0
        
        with self.driver.session(database=self.database) as session:
            with session.begin_transaction() as tx:
                for relation in relations:
                    cypher = """
                    MATCH (source:Entity {id: $source_id})
                    MATCH (target:Entity {id: $target_id})
                    MERGE (source)-[r:RELATES {id: $id, type: $type}]->(target)
                    SET r.description = $description,
                        r.properties = $properties,
                        r.source_papers = $source_papers,
                        r.confidence = $confidence,
                        r.created_at = $created_at,
                        r.updated_at = datetime()
                    """
                    
                    try:
                        tx.run(cypher, {
                            'id': relation.id,
                            'source_id': relation.source_id,
                            'target_id': relation.target_id,
                            'type': relation.type.value,
                            'description': relation.description,
                            'properties': json.dumps(relation.properties),
                            'source_papers': relation.source_papers,
                            'confidence': relation.confidence,
                            'created_at': relation.created_at
                        })
                        success_count += 1
                    except Exception as e:
                        print(f"插入关系 {relation.id} 失败: {e}")
        
        return success_count
    
    def add_relations_batch(self, relations: List[Relation]) -> int:
        """批量添加关系（别名方法，保持兼容性）"""
        return self.batch_insert_relations(relations)
    
    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """获取实体"""
        cypher = "MATCH (e:Entity {id: $id}) RETURN e"
        
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher, {'id': entity_id})
            record = result.single()
            if record:
                return dict(record['e'])
        return None
    
    def search_entities(self, name_pattern: str = None, entity_type: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """搜索实体"""
        conditions = []
        params = {'limit': limit}
        
        if name_pattern:
            conditions.append("e.name CONTAINS $name_pattern")
            params['name_pattern'] = name_pattern
        
        if entity_type:
            conditions.append("e.type = $entity_type")
            params['entity_type'] = entity_type
        
        where_clause = " AND ".join(conditions) if conditions else "true"
        cypher = f"MATCH (e:Entity) WHERE {where_clause} RETURN e LIMIT $limit"
        
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher, params)
            return [dict(record['e']) for record in result]
    
    def get_entity_relations(self, entity_id: str, direction: str = "both") -> List[Dict[str, Any]]:
        """获取实体的关系"""
        if direction == "outgoing":
            cypher = """
            MATCH (e:Entity {id: $id})-[r:RELATES]->(target:Entity)
            RETURN r, target
            """
        elif direction == "incoming":
            cypher = """
            MATCH (source:Entity)-[r:RELATES]->(e:Entity {id: $id})
            RETURN r, source
            """
        else:  # both
            cypher = """
            MATCH (e:Entity {id: $id})-[r:RELATES]-(other:Entity)
            RETURN r, other
            """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher, {'id': entity_id})
            relations = []
            for record in result:
                rel_data = dict(record['r'])
                other_entity = dict(record.get('target') or record.get('source') or record.get('other'))
                relations.append({
                    'relation': rel_data,
                    'connected_entity': other_entity
                })
            return relations
    
    def find_shortest_path(self, source_id: str, target_id: str, max_length: int = 5) -> List[Dict[str, Any]]:
        """查找最短路径"""
        cypher = """
        MATCH path = shortestPath((source:Entity {id: $source_id})-[*1..{max_length}]-(target:Entity {id: $target_id}))
        RETURN path
        """.format(max_length=max_length)
        
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher, {
                'source_id': source_id,
                'target_id': target_id
            })
            paths = []
            for record in result:
                path = record['path']
                path_data = {
                    'length': len(path.relationships),
                    'nodes': [dict(node) for node in path.nodes],
                    'relationships': [dict(rel) for rel in path.relationships]
                }
                paths.append(path_data)
            return paths
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取图数据库统计信息"""
        stats_queries = {
            'total_entities': "MATCH (e:Entity) RETURN count(e) as count",
            'total_relations': "MATCH ()-[r:RELATES]->() RETURN count(r) as count",
            'entity_types': "MATCH (e:Entity) RETURN e.type as type, count(e) as count ORDER BY count DESC",
            'relation_types': "MATCH ()-[r:RELATES]->() RETURN r.type as type, count(r) as count ORDER BY count DESC"
        }
        
        stats = {}
        with self.driver.session(database=self.database) as session:
            # 获取总数
            for key, query in stats_queries.items():
                if key in ['entity_types', 'relation_types']:
                    result = session.run(query)
                    stats[key] = {record['type']: record['count'] for record in result}
                else:
                    result = session.run(query)
                    record = result.single()
                    stats[key] = record['count'] if record else 0
        
        return stats
    
    def clear_database(self) -> bool:
        """清空数据库（谨慎使用）"""
        cypher = "MATCH (n) DETACH DELETE n"
        
        try:
            with self.driver.session(database=self.database) as session:
                session.run(cypher)
            print("✓ 数据库已清空")
            return True
        except Exception as e:
            print(f"清空数据库失败: {e}")
            return False
    
    def get_full_graph(self, limit: int = 200) -> Tuple[List[Dict], List[Dict]]:
        """获取整个图谱用于可视化"""
        entities_query = f"MATCH (n:Entity) RETURN n LIMIT {limit}"
        relations_query = f"MATCH (n:Entity)-[r:RELATES]->(m:Entity) RETURN n.name AS source_name, m.name AS target_name, r.type AS type LIMIT {limit}"

        entities = []
        relations = []

        with self.driver.session(database=self.database) as session:
            # Get entities
            result = session.run(entities_query)
            for record in result:
                entities.append(dict(record['n']))
            
            # Get relations
            result = session.run(relations_query)
            for record in result:
                relations.append(dict(record))
        
        return entities, relations

    def execute_custom_query(self, cypher: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """执行自定义Cypher查询"""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(cypher, parameters or {})
                return [dict(record) for record in result]
        except Exception as e:
            print(f"查询执行失败: {e}")
            return [] 