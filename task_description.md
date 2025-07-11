# Task Description

一、任务描述
本次工业技术文档多模态推理问答评测任务，具有三大挑战：（1）图片型原始文档识别：不可编辑的PDF文档难以准确识别数据，分辨率低，格式多样，内容复杂。（2）多模态信息融合需求：问题解答常需同时解析文本描述和技术图纸的多模态数据；（3）复杂化领域知识推理：答案常需通过图纸结构解析、模块功能理解或机械原理推导获得。

二、评测数据
本次评测以JSONL格式提供样例集和评测集，并提供原始PDF文档集。参赛队伍可将样例集数据可用于模型微调训练，并自行从中划分出验证集。本评测任务数据来源于国内发明专利开放数据。

三、数据样例
根据工业技术文档内容回答指定问题，初赛评测任务为单选题问答，复赛评测任务为开放题问答。

初赛样例数据如下：

```json
{
"id": "1be0009101baa0fe95338f9542XXXXX", 
"question": "根据文本信息，以下哪个描述符合该静电除尘器的特征？", 
"document": "CN100342976C.pdf", 
"options": ["A. 具有平行于外壳主轴线的垂直方向的片状沉积电极。", "B. 具有管状入口和出口，它们分别由3种不同圆锥形部分所构成", "C. 管状入口具有单个圆锥形部分，达到外壳直径的80至95%，剩余部分采用台阶形式。", "D. 主要用于液体的除尘"], 
"answer": "C"
}
```

复赛样例数据如下：

```json
{
"id": "48707b8d6e06e49882a35dc67f5XXXXX", 
"question": "在文件中第7页的图片中，部件4相对于部件5在图片中的位置关系是？", 
"document": "CN100342976C.pdf", 
"answer": "部件4位于部件5的左侧"
}
```

本次评测的工业技术文档为不可编辑的PDF文档，样例数据如下：

四、结果提交
测试结果文件须为JSONL格式，其中每条数据只需包含"question_id"和"answer"两个字段即可。禁止对模型回答进行人工修正。

初赛结果文件数据样例

```json
{
 "id": "1be0009101baa0fe95338f9542XXXXX",
 "answer": "C"
}
```

复赛结果文件数据样例

```json
{
 "id": "48707b8d6e06e49882a35dc67f5XXXXX",
 "answer": "部件4位于部件5的左侧"
}
```

五、评测指标
初赛评测指标为分类准确率（Acc），复赛评测指标为精确匹配准确率（EM）和F1分数。
