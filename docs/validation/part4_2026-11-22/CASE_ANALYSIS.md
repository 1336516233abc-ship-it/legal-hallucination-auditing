# Part 4 错误与修订案例清单

本文件是已有实验输出的定性案例整理，不是课程报告。未重新运行模型，未修改原预测、金标或已发布指标。

## 范围与结论边界

- 核对了两个模型 × 三个变体的全部 60 条输出，对应 10 个相同问题（ContractNLI 6、CUAD 4），不是 60 个独立样本。下方 9 个案例按不同问题类型选取，不是随机抽样，不能计算总体错误类型占比。
- 原始标签统计仍为 37/60 正确、57/60 格式通过。逐条另查引用范围、金标证据覆盖和最终展示答案；标签正确不保证引用正确或答案可用。
- 全部输出中有 3 条格式失败、11 条程序记录的 grounding_error。重复样本跨变体保留，不当作独立失败事件。
- 20 条 Full 输出中实际修订 11 次：标签错→对 1 次、错→错 4 次、对→对 6 次、对→错 0 次。另有 2 次修订引入新的越界引用。唯一标签纠正案例仍有引用指向错误，不能把 1/11 或 1/5 宣称为独立幻觉修订成功率。
- 定性解释由 AI 辅助核对金标与合同原文，尚未经过双人独立盲评；独立幻觉率、完整性、弃权精确度和完整修订成功率均保持未评估。neutral/absent 不自动等于弃权。
- 字符区间为处理后合同 context 的 Python 字符下标 [start,end)，不是 PDF 页码。JSON 摘录均校验等于原文切片；本页展示仅去除行尾空白。完整逐条索引及输入 SHA256 见 [case_analysis.json](case_analysis.json)。

## 案例索引

|编号|模型 / 数据集|主要观察|
|---|---|---|
|[C01](#c01)|llama3 / contractnli|标签修正，但引用仍指错证据|
|[C02](#c02)|mistral / contractnli|解释已经否定命题，标签却仍为 neutral|
|[C03](#c03)|llama3 / cuad|重检索找到签署日期，答案仍为 absent|
|[C04](#c04)|mistral / cuad|重检索覆盖续期通知条款，仍漏掉 180 天|
|[C05](#c05)|mistral / contractnli|输出 Label: true，任务协议解析失败|
|[C06](#c06)|llama3 / cuad|占位符逐字匹配通过，但没有回答目标问题|
|[C07](#c07)|llama3 / contractnli|修订引入越界引用，正确标签下的答案被清空|
|[C08](#c08)|mistral / cuad|标签保持正确，修订却添加大量不存在的引用|
|[C09](#c09)|llama3 / contractnli|初始引用越界使验证和修订均未执行|

<a id="c01"></a>

## C01 · 标签修正，但引用仍指错证据

- 样本：`contractnli_test_523_nda-8`
- 问题：Is this statement true under the contract: "Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information."?
- 金标：`entailment`；初始 → 最终：`contradiction` → `entailment`；执行修订：是。
- 原始输出：[llama3/contractnli/full 第 4 行](agent_ablation/llama3/contractnli/full/predictions.jsonl#L4)。

**初始模型响应**

```text
Label: contradiction
Answer: The contract does not require the Receiving Party to notify the Disclosing Party in case the Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information. Instead, it states that the Receiving Party will "immediately notify the Company of the existence, terms and circumstances surrounding such request" and "consult with the Company on the advisability of taking legally available steps to resist or narrow such request" (Evidence [2]).
```

**最终模型响应**

```text
Label: entailment
Answer: The contract requires the Receiving Party to "immediately notify the Company of the existence, terms and circumstances surrounding such request" and "consult with the Company on the advisability of taking legally available steps to resist or narrow such request" if the Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information (Evidence [2]).
```

最终展示：保留模型答案。

|阶段|输入证据块数|金标区间覆盖率|模型引用编号|
|---|---:|---|---|
|初始|7|1.0|[2]|
|最终|3|1.0|[2]|

**合同原文与核对**

数据集金标证据，`[4181,4495)`：

```text
In the event that you or any of your representatives received a request or are required (by deposition, interrogatory, request for documents, subpoena, civil investigative demand or similar process) to disclose all or any part of the Evaluation Materials, you or your representatives, as the case may be, agree to
```

数据集金标证据，`[4495,4598)`：

```text
(i) immediately notify the Company of the existence, terms and circumstances surrounding such request,
```

初始 contradiction 与模型自己引用的通知义务相冲突；修订后 entailment 与金标和合同一致。但重检索使通知义务从证据 [2] 移到 [1]，最终仍写 Evidence [2]。只能称标签纠正，不能称完整的证据引用修复。编号保留是观察结果，其内部原因未通过新增实验验证。

**同一样本的变体对照**

|变体|最终标签|标签正确|格式通过|引用/抽取检查错误|
|---|---|---|---|---|
|full|entailment|是|是|无程序标记；不代表语义已验证|
|no_verification|contradiction|否|是|无程序标记；不代表语义已验证|
|no_revision|contradiction|否|是|无程序标记；不代表语义已验证|

后续检查方向：检查修订后的引用是否支持对应断言，而不只检查编号是否越界。


<a id="c02"></a>

## C02 · 解释已经否定命题，标签却仍为 neutral

- 样本：`contractnli_test_493_nda-2`
- 问题：Is this statement true under the contract: "Confidential Information shall only include technical information."?
- 金标：`contradiction`；初始 → 最终：`neutral` → `neutral`；执行修订：是。
- 原始输出：[mistral/contractnli/full 第 2 行](agent_ablation/mistral/contractnli/full/predictions.jsonl#L2)。

**初始模型响应**

```text
Label: neutral
Answer: The contract does not explicitly state that Confidential Information shall only include technical information. [1] However, the definition of Confidential Information in the contract includes various types of information such as trade secrets, know-how, inventions, designs, methods, techniques, systems, processes, software programs, works of authorship, business plans, customer lists, projects, plans, pricing, proposals, and any other information. [2] Therefore, while technical information is included in the definition, it is not the only type of Confidential Information.
```

**最终模型响应**

```text
Label: neutral
Answer: The contract does not state that Confidential Information only includes technical information [1]. While technical information is included in the definition of Confidential Information, other types of information are also covered [1].
```

最终展示：保留模型答案。

|阶段|输入证据块数|金标区间覆盖率|模型引用编号|
|---|---:|---|---|
|初始|4|1.0|[1, 2]|
|最终|3|1.0|[1, 1]|

**合同原文与核对**

数据集金标证据，`[936,1276)`：

```text
A. "Confidential Information" of a party shall mean any trade secrets, know-how, inventions, products, designs, methods, techniques, systems, processes, software programs, works of authorship, business plans, customer lists, projects, plans, pricing, proposals and any other information which a party discloses to the Recipient Party that:
```

问题含 only technical information。合同定义明确列出 business plans、customer lists、pricing 等范围，金标为 contradiction。初始解释和修订解释均承认不止技术信息，但标签始终 neutral；这不是缺少金标证据，而是标签与解释不一致。

**同一样本的变体对照**

|变体|最终标签|标签正确|格式通过|引用/抽取检查错误|
|---|---|---|---|---|
|full|neutral|否|是|无程序标记；不代表语义已验证|
|no_verification|neutral|否|是|无程序标记；不代表语义已验证|
|no_revision|neutral|否|是|无程序标记；不代表语义已验证|

后续检查方向：核对 only、all 等限定词及标签与解释的一致性。


<a id="c03"></a>

## C03 · 重检索找到签署日期，答案仍为 absent

- 样本：`cuad_test_HPILHOLDING_01_07_2015-EX-99.1-COOPERATION AGREEMENT__Agreement Date`
- 问题：Highlight the parts (if any) of this contract related to "Agreement Date" that should be reviewed by a lawyer. Details: The date of the contract
- 金标：`present`；初始 → 最终：`absent` → `absent`；执行修订：是。
- 原始输出：[llama3/cuad/full 第 2 行](agent_ablation/llama3/cuad/full/predictions.jsonl#L2)。

**初始模型响应**

```text
{"label":"absent","extracted_spans":[],"citations":[]}
```

**最终模型响应**

```text
{"label":"absent","extracted_spans":[],"citations":[]}
```

最终展示：合法 absent 的空抽取。

|阶段|输入证据块数|金标区间覆盖率|模型引用编号|
|---|---:|---|---|
|初始|3|0.0|[]|
|最终|3|1.0|[]|

**合同原文与核对**

数据集金标证据，`[3560,3584)`：

```text
5th day of January, 2015
```

签署日期上下文，`[3504,3605)`：

```text
THIS COOPERATION AGREEMENT ("Agreement") is signed this 5th day of January, 2015 (the "Closing Date")
```

初始提示没有覆盖日期金标；重检索后证据 [3] 覆盖签署日期，标签仍为 absent。该样本同时体现初始检索遗漏和重检索后的生成遗漏，不能只归因于检索。Mistral 的三个变体均抽取出该日期，可作为同一问题的对照。

**同一样本的变体对照**

|变体|最终标签|标签正确|格式通过|引用/抽取检查错误|
|---|---|---|---|---|
|full|absent|否|是|无程序标记；不代表语义已验证|
|no_verification|absent|否|是|无程序标记；不代表语义已验证|
|no_revision|absent|否|是|无程序标记；不代表语义已验证|

后续检查方向：分别检查初始与修订提示中的目标证据，再评估模型是否使用它。


<a id="c04"></a>

## C04 · 重检索覆盖续期通知条款，仍漏掉 180 天

- 样本：`cuad_test_PACIRA PHARMACEUTICALS, INC. - A_R STRATEGIC LICENSING, DISTRIBUTION AND MARKETING AGREEMENT __Notice Period To Terminate Renewal`
- 问题：Highlight the parts (if any) of this contract related to "Notice Period To Terminate Renewal" that should be reviewed by a lawyer. Details: What is the notice period required to terminate renewal?
- 金标：`present`；初始 → 最终：`absent` → `absent`；执行修订：是。
- 原始输出：[mistral/cuad/full 第 4 行](agent_ablation/mistral/cuad/full/predictions.jsonl#L4)。

**初始模型响应**

```text
{"label":"absent","extracted_spans":[],"citations":[]}
```

**最终模型响应**

```text
Based on the provided evidence, there is no explicit clause stating the notice period to terminate a renewal. The contract does mention a notice period for termination by either party under certain circumstances (Section 16.1), but it does not specify a notice period for terminating a renewal term. Therefore, the label should be "absent" and the answer should be:

{"label":"absent","extracted_spans":[],"citations":[]}
```

最终展示：合法 absent 的空抽取。

|阶段|输入证据块数|金标区间覆盖率|模型引用编号|
|---|---:|---|---|
|初始|5|0.0|[]|
|最终|3|1.0|[]|

**合同原文与核对**

数据集金标证据，`[94251,94488)`：

```text
Notwithstanding the foregoing, this  Agreement can be terminated by EKR at the end of the Initial Term by delivery of written notice to PPI at least one hundred eighty (180)  days prior to the end of the Initial Term or any renewal term.
```

初始提示未覆盖金标，重检索后证据 [2] 已含 at least one hundred eighty (180) days 及 any renewal term。最终仍说明没有终止续期的明确通知期，预测 absent，与金标 present 不符。

**同一样本的变体对照**

|变体|最终标签|标签正确|格式通过|引用/抽取检查错误|
|---|---|---|---|---|
|full|absent|否|是|无程序标记；不代表语义已验证|
|no_verification|absent|否|是|无程序标记；不代表语义已验证|
|no_revision|absent|否|是|无程序标记；不代表语义已验证|

后续检查方向：结合完整条件与 renewal term 审阅条款，避免把存在的续期条件漏判为不存在。


<a id="c05"></a>

## C05 · 输出 Label: true，任务协议解析失败

- 样本：`contractnli_test_523_nda-8`
- 问题：Is this statement true under the contract: "Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information."?
- 金标：`entailment`；初始 → 最终：`invalid` → `invalid`；执行修订：否。
- 原始输出：[mistral/contractnli/full 第 4 行](agent_ablation/mistral/contractnli/full/predictions.jsonl#L4)。

**初始模型响应**

```text
Label: true
Answer: The contract requires the Receiving Party to notify the Disclosing Party if they are required by law, regulation, or judicial process to disclose any Confidential Information [1].
```

**最终模型响应**

```text
Label: true
Answer: The contract requires the Receiving Party to notify the Disclosing Party if they are required by law, regulation, or judicial process to disclose any Confidential Information [1].
```

最终展示：因解析或引用检查错误而清空。

|阶段|输入证据块数|金标区间覆盖率|模型引用编号|
|---|---:|---|---|
|初始|7|1.0|[]|
|最终|7|1.0|[]|

**合同原文与核对**

数据集金标证据，`[4181,4495)`：

```text
In the event that you or any of your representatives received a request or are required (by deposition, interrogatory, request for documents, subpoena, civil investigative demand or similar process) to disclose all or any part of the Evaluation Materials, you or your representatives, as the case may be, agree to
```

数据集金标证据，`[4495,4598)`：

```text
(i) immediately notify the Company of the existence, terms and circumstances surrounding such request,
```

初始原响应使用 Label: true，而 ContractNLI 只接受 entailment、contradiction、neutral。三个变体均返回相同非法标签并计错；Full 因解析失败跳过验证和修订，最终展示答案为空。即使说明文字提到通知义务，也不能改记为格式通过或标签正确。

**同一样本的变体对照**

|变体|最终标签|标签正确|格式通过|引用/抽取检查错误|
|---|---|---|---|---|
|full|invalid|否|否|无程序标记；不代表语义已验证|
|no_verification|invalid|否|否|无程序标记；不代表语义已验证|
|no_revision|invalid|否|否|无程序标记；不代表语义已验证|

后续检查方向：后续单独验证结构化输出或格式恢复策略；不能在结果汇总时事后修改原预测。


<a id="c06"></a>

## C06 · 占位符逐字匹配通过，但没有回答目标问题

- 样本：`cuad_test_N2KINC_10_16_1997-EX-10.16-SPONSORSHIP AGREEMENT__Notice Period To Terminate Renewal`
- 问题：Highlight the parts (if any) of this contract related to "Notice Period To Terminate Renewal" that should be reviewed by a lawyer. Details: What is the notice period required to terminate renewal?
- 金标：`absent`；初始 → 最终：`present` → `present`；执行修订：否。
- 原始输出：[llama3/cuad/full 第 3 行](agent_ablation/llama3/cuad/full/predictions.jsonl#L3)。

**初始模型响应**

```text
{"label":"present","extracted_spans":["[****]"],"citations":[2]}
```

**最终模型响应**

```text
{"label":"present","extracted_spans":["[****]"],"citations":[2]}
```

最终展示：保留模型答案。

|阶段|输入证据块数|金标区间覆盖率|模型引用编号|
|---|---:|---|---|
|初始|5|无金标区间，不计|[2]|
|最终|5|无金标区间，不计|[2]|

**合同原文与核对**

续期谈判上下文；非金标通知期，`[6380,7203)`：

```text
ii) Commencing not later than [****] prior to the expiration                   of the term of the Agreement, Excite will negotiate with                   Sponsor in good faith with respect to the terms and conditions                   under which this Agreement would be renewed. Excite will                   negotiate exclusively with Sponsor for the next [****] in good                   faith effort to negotiate and execute a written sponsorship                   renewal agreement. If, [****] prior to the expiration of the                   term of the Agreement, the parties have not entered into a                   written sponsorship renewal agreement. Excite may enter into                   negotiations with any third party with respect to retail music                   store sponsorships of the Excite Site.
```

金标为 absent。模型却将脱敏占位符 [****] 当作 present 的抽取答案。该字符串确实出现在输入中，因此逐字匹配通过；Verifier 的 supported 理由引用的是广告展示次数（Impressions），并未证明终止续期通知期。Full 因未触发检测而没有修订。这里可确认局部任务相关性检查不足，不能据此估计总体幻觉率。

**同一样本的变体对照**

|变体|最终标签|标签正确|格式通过|引用/抽取检查错误|
|---|---|---|---|---|
|full|present|否|是|无程序标记；不代表语义已验证|
|no_verification|present|否|是|无程序标记；不代表语义已验证|
|no_revision|present|否|是|无程序标记；不代表语义已验证|

后续检查方向：验证器应检查所摘内容是否回答目标条款问题，而不只检查字符串是否出现。


<a id="c07"></a>

## C07 · 修订引入越界引用，正确标签下的答案被清空

- 样本：`contractnli_test_53_nda-18`
- 问题：Is this statement true under the contract: "Receiving Party shall not solicit some of Disclosing Party's representatives."?
- 金标：`neutral`；初始 → 最终：`neutral` → `neutral`；执行修订：是。
- 原始输出：[llama3/contractnli/full 第 5 行](agent_ablation/llama3/contractnli/full/predictions.jsonl#L5)。

**初始模型响应**

```text
Label: neutral
Answer: The document does not specify that the Receiving Party shall not solicit some of Disclosing Party's representatives.
```

**最终模型响应**

```text
Label: neutral
Answer: The document does not specify that the Receiving Party shall not solicit some of Disclosing Party's representatives.

The original question asks if the Receiving Party shall not solicit some of Disclosing Party's representatives. However, the contract does not explicitly state this. The closest relevant clause is clause 18, which states that "Nothing in this Agreement is intended to or shall operate to create a partnership or joint venture of any kind between the Parties (or any of them), or to authorise any Party to act as agent for another, and no Party shall have authority to act in the name or on behalf of or otherwise to bind any other in any way." [18] This clause does not specifically prohibit the Receiving Party from soliciting representatives of the Disclosing Party. Therefore, the answer is neutral, as the document does not provide sufficient information to determine whether the Receiving Party shall not solicit some of Disclosing Party's representatives.
```

最终展示：因解析或引用检查错误而清空。

|阶段|输入证据块数|金标区间覆盖率|模型引用编号|
|---|---:|---|---|
|初始|6|无金标区间，不计|[]|
|最终|3|无金标区间，不计|[18]|

**合同原文与核对**

合同第 18 条；用于解释编号混淆，不是招揽限制证据，`[13542,13861)`：

```text
18. Nothing in this Agreement is intended to or shall operate to create a partnership or joint venture of any kind between the Parties (or any of them), or to authorise any Party to act as agent for another, and no Party shall have authority to act in the name or on behalf of or otherwise to bind any other in any way.
```

初始和最终标签均为 neutral，与金标一致。修订响应添加了 [18]，但最终输入只有 3 个证据块。原合同确有第 18 条，内容涉及非合伙和代理关系；它既不是证据编号 [18]，也不是禁止招揽条款。范围检查据此标记 grounding_error，展示答案为空。这是新增引用错误，不能称修订成功。neutral 本身不作为人工弃权标签。

**同一样本的变体对照**

|变体|最终标签|标签正确|格式通过|引用/抽取检查错误|
|---|---|---|---|---|
|full|neutral|是|是|answer cited evidence outside the model input|
|no_verification|neutral|是|是|无程序标记；不代表语义已验证|
|no_revision|neutral|是|是|无程序标记；不代表语义已验证|

后续检查方向：将合同条款号与提示证据编号分开，并检查修订是否引入新错误。


<a id="c08"></a>

## C08 · 标签保持正确，修订却添加大量不存在的引用

- 样本：`cuad_test_ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT VENTURE AGREEMENT__Notice Period To Terminate Renewal`
- 问题：Highlight the parts (if any) of this contract related to "Notice Period To Terminate Renewal" that should be reviewed by a lawyer. Details: What is the notice period required to terminate renewal?
- 金标：`absent`；初始 → 最终：`absent` → `absent`；执行修订：是。
- 原始输出：[mistral/cuad/full 第 1 行](agent_ablation/mistral/cuad/full/predictions.jsonl#L1)。

**初始模型响应**

```text
{"label":"absent","extracted_spans":[],"citations":[]}
```

**最终模型响应**

```text
{"label":"absent","extracted_spans":[],"citations":[1,2,3,4,5,11,12,13,14,15,16]}
```

最终展示：因解析或引用检查错误而清空。

|阶段|输入证据块数|金标区间覆盖率|模型引用编号|
|---|---:|---|---|
|初始|5|无金标区间，不计|[]|
|最终|3|无金标区间，不计|[1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16]|

**合同原文与核对**

合资期限和延期约定；金标 absent 没有目标证据区间，`[5991,6263)`：

```text
8.   Term of Joint Venture. The Joint Venture shall commence on the 1st of      March, 2003, and shall be effective until February 28, 2004 unless extended      by written agreement of the Joint Venturers not less than thirty (30) days      prior to scheduled termination.
```

初始 absent 与金标一致且无引用错误。修订后仍为 absent，但加入 [1,2,3,4,5,11,12,13,14,15,16]；最终只有 3 个证据块，8 个编号越界。absent 的空抽取原本是合法结构，不能仅因答案空白判错；这里的确切新增问题是引用越界。

**同一样本的变体对照**

|变体|最终标签|标签正确|格式通过|引用/抽取检查错误|
|---|---|---|---|---|
|full|absent|是|是|answer cited evidence outside the model input|
|no_verification|absent|是|是|无程序标记；不代表语义已验证|
|no_revision|absent|是|是|无程序标记；不代表语义已验证|

后续检查方向：分别统计标签变化、合法空抽取与新增引用错误，避免把它们混成一个成功率。


<a id="c09"></a>

## C09 · 初始引用越界使验证和修订均未执行

- 样本：`contractnli_test_4_nda-4`
- 问题：Is this statement true under the contract: "Receiving Party shall not use any Confidential Information for any purpose other than the purposes stated in Agreement."?
- 金标：`entailment`；初始 → 最终：`entailment` → `entailment`；执行修订：否。
- 原始输出：[llama3/contractnli/full 第 3 行](agent_ablation/llama3/contractnli/full/predictions.jsonl#L3)。

**初始模型响应**

```text
Label: entailment
Answer: The Recipient undertakes not to use the Confidential Information for any purpose except the Purpose, without first obtaining the written agreement of the Discloser. [2]
```

**最终模型响应**

```text
Label: entailment
Answer: The Recipient undertakes not to use the Confidential Information for any purpose except the Purpose, without first obtaining the written agreement of the Discloser. [2]
```

最终展示：因解析或引用检查错误而清空。

|阶段|输入证据块数|金标区间覆盖率|模型引用编号|
|---|---:|---|---|
|初始|1|1.0|[2]|
|最终|1|1.0|[2]|

**合同原文与核对**

数据集金标证据，`[444,611)`：

```text
2. The Recipient undertakes not to use the Confidential Information for any purpose except the Purpose, without first obtaining the written agreement of the Discloser.
```

合同第 2 条支持用途限制，预测 entailment 也正确。但模型使用 [2]，而输入只有 1 个证据块；grounding_error 导致验证和修订跳过，最终展示为空。与 C07 不同，这个错误在初始阶段已出现，不能归因于修订器。

**同一样本的变体对照**

|变体|最终标签|标签正确|格式通过|引用/抽取检查错误|
|---|---|---|---|---|
|full|entailment|是|是|answer cited evidence outside the model input|
|no_verification|entailment|是|是|answer cited evidence outside the model input|
|no_revision|entailment|是|是|answer cited evidence outside the model input|

后续检查方向：优先检查证据编号与合同条款号的混用；再单独评估是否需要格式或引用恢复分支。

## 全部 Full 输出索引

保留正确、错误及格式失败输出，便于核查选例范围。

|模型 / 数据集|样本（可打开完整轨迹）|金标|初始 → 最终|修订|grounding_error|
|---|---|---|---|---|---|
|llama3 / contractnli|[41_nda-1](agent_ablation/llama3/contractnli/full/predictions.jsonl#L1)|contradiction|contradiction → contradiction|否|无程序标记|
|llama3 / contractnli|[493_nda-2](agent_ablation/llama3/contractnli/full/predictions.jsonl#L2)|contradiction|contradiction → contradiction|否|无程序标记|
|llama3 / contractnli|[4_nda-4](agent_ablation/llama3/contractnli/full/predictions.jsonl#L3)|entailment|entailment → entailment|否|有|
|llama3 / contractnli|[523_nda-8](agent_ablation/llama3/contractnli/full/predictions.jsonl#L4)|entailment|contradiction → entailment|是|无程序标记|
|llama3 / contractnli|[53_nda-18](agent_ablation/llama3/contractnli/full/predictions.jsonl#L5)|neutral|neutral → neutral|是|有|
|llama3 / contractnli|[59_nda-13](agent_ablation/llama3/contractnli/full/predictions.jsonl#L6)|neutral|neutral → neutral|是|无程序标记|
|llama3 / cuad|[ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT VENTURE AGREEMENT__Notice Period To Terminate Renewal](agent_ablation/llama3/cuad/full/predictions.jsonl#L1)|absent|present → present|否|有|
|llama3 / cuad|[HPILHOLDING_01_07_2015-EX-99.1-COOPERATION AGREEMENT__Agreement Date](agent_ablation/llama3/cuad/full/predictions.jsonl#L2)|present|absent → absent|是|无程序标记|
|llama3 / cuad|[N2KINC_10_16_1997-EX-10.16-SPONSORSHIP AGREEMENT__Notice Period To Terminate Renewal](agent_ablation/llama3/cuad/full/predictions.jsonl#L3)|absent|present → present|否|无程序标记|
|llama3 / cuad|[PACIRA PHARMACEUTICALS, INC. - A_R STRATEGIC LICENSING, DISTRIBUTION AND MARKETING AGREEMENT __Notice Period To Terminate Renewal](agent_ablation/llama3/cuad/full/predictions.jsonl#L4)|present|present → present|否|有|
|mistral / contractnli|[41_nda-1](agent_ablation/mistral/contractnli/full/predictions.jsonl#L1)|contradiction|neutral → neutral|是|无程序标记|
|mistral / contractnli|[493_nda-2](agent_ablation/mistral/contractnli/full/predictions.jsonl#L2)|contradiction|neutral → neutral|是|无程序标记|
|mistral / contractnli|[4_nda-4](agent_ablation/mistral/contractnli/full/predictions.jsonl#L3)|entailment|entailment → entailment|否|无程序标记|
|mistral / contractnli|[523_nda-8](agent_ablation/mistral/contractnli/full/predictions.jsonl#L4)|entailment|invalid → invalid|否|无程序标记|
|mistral / contractnli|[53_nda-18](agent_ablation/mistral/contractnli/full/predictions.jsonl#L5)|neutral|neutral → neutral|是|无程序标记|
|mistral / contractnli|[59_nda-13](agent_ablation/mistral/contractnli/full/predictions.jsonl#L6)|neutral|neutral → neutral|是|无程序标记|
|mistral / cuad|[ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT VENTURE AGREEMENT__Notice Period To Terminate Renewal](agent_ablation/mistral/cuad/full/predictions.jsonl#L1)|absent|absent → absent|是|有|
|mistral / cuad|[HPILHOLDING_01_07_2015-EX-99.1-COOPERATION AGREEMENT__Agreement Date](agent_ablation/mistral/cuad/full/predictions.jsonl#L2)|present|present → present|否|无程序标记|
|mistral / cuad|[N2KINC_10_16_1997-EX-10.16-SPONSORSHIP AGREEMENT__Notice Period To Terminate Renewal](agent_ablation/mistral/cuad/full/predictions.jsonl#L3)|absent|absent → absent|是|无程序标记|
|mistral / cuad|[PACIRA PHARMACEUTICALS, INC. - A_R STRATEGIC LICENSING, DISTRIBUTION AND MARKETING AGREEMENT __Notice Period To Terminate Renewal](agent_ablation/mistral/cuad/full/predictions.jsonl#L4)|present|absent → absent|是|无程序标记|

## 后续优先级

1. 检查引用编号：同时覆盖编号越界和编号合法但指错内容两类；重新检索后的编号需要重新对应。
2. 检查任务相关性：逐字摘录或 verifier 的 supported 标志不能替代是否回答目标问题。
3. 检查标签与说明一致性，以及重检索拿到目标条款后的漏答。
4. 单独验证格式恢复策略，保留原始失败记录，不能事后改写已公布结果。
以上是由案例提出的检查方向，本次未更改 Agent 实现，也未运行修复后的对照实验。

## 复现与完整性

在仓库根目录执行（仅标准库，不加载模型）：

```bash
python3 docs/validation/part4_2026-11-22/build_case_analysis.py
```

需要原 processed test 文件与本目录 12 份 JSONL。脚本核对模型间固定样本、数据 SHA256、全部 60 条最终评分、初始/最终证据切片、引用范围及展示抑制条件；同时验证上述案例特有的标签变化与证据位置。
原实验主表：[RESULTS.md](RESULTS.md)。独立评审方案：[ANSWER_SUPPORT_PROTOCOL.md](../../ANSWER_SUPPORT_PROTOCOL.md)。
