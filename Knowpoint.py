import dashscope
from dashscope import Generation
import json
from tqdm import tqdm


# ------------------------
# 设置 API Key
# ------------------------
dashscope.api_key = "sk-3e784a5991b44e4ea8edb76de2d1009b"

# ------------------------
# 选择模型
# ------------------------
MODEL_NAME = "qwen2.5-14b-instruct"

# ------------------------
# Prompt 模板（已修复）
# ------------------------
EXTRACTION_PROMPT = """
你是一个专业的知识抽取系统。给你一段文化、建筑、美食、民族或地理相关的文本，
请抽取所有 (头实体, 关系, 尾实体) 三元组。

# 实体类型（严格从文本中识别，不得生成文本外实体）
1. 地理实体: 省份、城市、地区、山脉、河流、湖泊
2. 建筑实体: 历史建筑、现代建筑、桥梁、寺庙、石窟
3. 人物实体: 历史人物、现代人物、文学人物、神话人物
4. 民族与文化实体: 民族、语言、文字
5. 食品与饮食实体: 菜肴、小吃、饮品
6. 服饰实体: 传统服饰、配饰
7. 艺术与表演实体: 戏曲、舞蹈、乐器
8. 节日与习俗实体: 传统节日、习俗
9. 历史事件实体: 历史事件、朝代

# 关系类型（只能从以下词汇中选择）
空间关系: 位于、流经、接壤  
时间关系: 发生于、建于、活跃于、始于、盛于、终于  
文化归属: 属于、由...实践、由...使用  
影响关系: 受...影响、源于、演变为  
组成关系: 包含、由...制成、由...组成  
功能关系: 用于、作为、代表  
人物关系: 由...创造、师从、与...同时代  
属性关系: 材质为、颜色为、特点为、用途为

# 抽取规则
1. 实体必须来源于文本，不要创造新实体
2. 关系词必须从上述定义中选择最合适的一个
3. 重点关注能回答"是什么、在哪里、什么时候、有什么特点"的知识
4. 三元组应该完整、准确、有意义

# 文本内容
{knowledge_text}

# 输出要求
请输出JSON数组，每个三元组格式：
{{
  "head": "头实体",
  "relation": "关系词", 
  "tail": "尾实体",
  "head_type": "实体类型",
  "tail_type": "实体类型"
}}

请直接输出JSON数组，不要有其他文字：
"""

# ------------------------
# 修复后的三元组抽取函数
# ------------------------
def extract_triples(knowledge_text):
    """从知识文本中抽取三元组"""
    if not knowledge_text or len(knowledge_text.strip()) < 5:  # 降低长度限制
        print("📝 文本过短，跳过")
        return []

    prompt = EXTRACTION_PROMPT.format(knowledge_text=knowledge_text)

    try:
        response = Generation.call(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            result_format="message"
        )

        content = response["output"]["choices"][0]["message"]["content"]
        print(f"✅ 成功获取响应，长度: {len(content)}")

        # 尝试解析 JSON
        try:
            # 先清理内容
            cleaned_content = content.strip()

            # 移除代码块标记
            if cleaned_content.startswith('```json'):
                cleaned_content = cleaned_content[7:].strip()
            if cleaned_content.startswith('```'):
                cleaned_content = cleaned_content[3:].strip()
            if cleaned_content.endswith('```'):
                cleaned_content = cleaned_content[:-3].strip()

            # 直接解析
            triples = json.loads(cleaned_content)
            print(f"🎯 成功解析 {len(triples)} 个三元组")
            return triples

        except json.JSONDecodeError as e:
            print(f"⚠ JSON 解析失败，尝试修复...")
            print(f"原始内容前200字符: {content[:200]}")

            # 尝试提取JSON部分
            start_idx = content.find('[')
            end_idx = content.rfind(']')

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx+1]
                try:
                    triples = json.loads(json_str)
                    print(f"🔧 修复后解析 {len(triples)} 个三元组")
                    return triples
                except:
                    pass

            print(f"❌ 无法解析JSON: {e}")
            return []

    except Exception as e:
        print(f"❌ API调用失败: {e}")
        return []

# ------------------------
# 修复后的主流程
# ------------------------
def build_kg(data_path, output_path, experiment_mode=False, max_items=100):
    """
    构建知识图谱
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 实验模式：只处理前N条数据
    if experiment_mode:
        data = data[:max_items]
        print(f"🔬 实验模式：只处理前{max_items}条数据")

    kg_triples = []
    processed_count = 0
    success_count = 0
    error_count = 0

    for i, item in enumerate(tqdm(data, desc="抽取三元组中")):
        kp = item.get("Knowledge_Point", "").strip()
        if not kp:
            print(f"📝 第{i+1}条数据Knowledge_Point为空，跳过")
            continue

        processed_count += 1
        print(f"\n--- 处理第{processed_count}条数据 ---")
        print(f"知识内容: {kp[:100]}...")

        try:
            triples = extract_triples(kp)

            # 修复关键问题：空列表也应该被处理
            if triples is not None:  # 只要不是None就处理
                # 为每个三元组添加来源信息
                for triple in triples:
                    # 确保三元组有基本结构
                    if all(key in triple for key in ['head', 'relation', 'tail']):
                        triple['source_id'] = i
                        triple['source_text'] = kp[:100] + "..." if len(kp) > 100 else kp
                        # 确保有实体类型字段
                        if 'head_type' not in triple:
                            triple['head_type'] = '未知'
                        if 'tail_type' not in triple:
                            triple['tail_type'] = '未知'

                kg_triples.extend(triples)
                success_count += 1
                print(f"✅ 成功添加 {len(triples)} 个三元组")
            else:
                error_count += 1
                print(f"❌ 抽取失败")

        except Exception as e:
            print(f"❌ 处理第{processed_count}条数据时出错: {e}")
            error_count += 1
            continue

    print(f"\n📊 最终统计:")
    print(f"处理数据条数: {processed_count}")
    print(f"成功抽取: {success_count} 条")
    print(f"抽取失败: {error_count} 条")
    print(f"总三元组数: {len(kg_triples)} 条")

    # 保存结果
    if kg_triples:
        result = {
            "metadata": {
                "total_processed": processed_count,
                "success_count": success_count,
                "error_count": error_count,
                "total_triples": len(kg_triples)
            },
            "triples": kg_triples
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"💾 成功保存到: {output_path}")

        # 预览
        print(f"\n👀 前5个三元组预览:")
        for i, triple in enumerate(kg_triples[:5]):
            print(f"{i+1}. {triple['head']} --{triple['relation']}--> {triple['tail']}")
    else:
        print("❌ 没有抽取到任何三元组，不保存文件")

    return kg_triples

# ------------------------
# 使用示例
# ------------------------
if __name__ == "__main__":
    input_file = r"D:\PyCharm\VQA\Data\multimodal_qa.json"  # 替换为您的实际路径
    output_file = "culture_kg_raw.json"

    # 先测试前10条
    build_kg(input_file, output_file, experiment_mode=True, max_items=10)