# 20 个可回查案例

按确定性标志选择，用于排查与说明，不是随机错误率样本，也不是人工幻觉标注。每个完整记录含合同全文、金标证据、提示证据和已有内部验证记录。

## 1. llama3 / contractnli / Direct

样本：`contractnli_test_117_nda-8`。 [完整记录](analysis/CASES.jsonl#L1)

问题：Is this statement true under the contract: "Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information."?

金标：`entailment`；终答标签：`contradiction`；初答标签：`None`。

自动选取标志：`gold_covered_but_label_wrong`。

最终可展示答案：

```text
The contract does not contain the statement "Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information."
```

## 2. llama3 / contractnli / RAG

样本：`contractnli_test_117_nda-15`。 [完整记录](analysis/CASES.jsonl#L2)

问题：Is this statement true under the contract: "Agreement shall not grant Receiving Party any right to Confidential Information."?

金标：`entailment`；终答标签：`entailment`；初答标签：`None`。

自动选取标志：`out_of_range_citation`。

最终可展示答案：

```text
The contract states that "ALL CONFIDENTIAL INFORMATION IS PROVIDED 'AS IS'. THE DISCLOSING PARTY MAKES NO WARRANTIES, EXPRESS, IMPLIED OR OTHERWISE, REGARDING NON-INFRINGEMENT OF THIRD PARTY RIGHTS OR ITS ACCURACY, COMPLETENESS OR PERFORMANCE." [3] This implies that the Disclosing Party does not grant any rights to the Recipient regarding the Confidential Information, including non-infringement of third-party rights.
```

## 3. llama3 / contractnli / RAG

样本：`contractnli_test_38_nda-15`。 [完整记录](analysis/CASES.jsonl#L3)

问题：Is this statement true under the contract: "Agreement shall not grant Receiving Party any right to Confidential Information."?

金标：`entailment`；终答标签：`invalid`；初答标签：`None`。

自动选取标志：`format_failure`。

最终可展示答案：

```text
（无可展示答案；请结合完整记录中的失败状态或合法空抽取判断）
```

## 4. llama3 / contractnli / Agent

样本：`contractnli_test_117_nda-1`。 [完整记录](analysis/CASES.jsonl#L4)

问题：Is this statement true under the contract: "All Confidential Information shall be expressly identified by the Disclosing Party."?

金标：`contradiction`；终答标签：`entailment`；初答标签：`contradiction`。

自动选取标志：`revision_correct_to_wrong`。

最终可展示答案：

```text
The contract states that the Disclosing Party "may, but is not required to, mark written Confidential Information with the legend 'Confidential' or an equivalent designation" [2, section 1], which implies that the Disclosing Party is not obligated to provide explicit identification of the Confidential Information.
```

## 5. llama3 / contractnli / Agent

样本：`contractnli_test_117_nda-5`。 [完整记录](analysis/CASES.jsonl#L5)

问题：Is this statement true under the contract: "Receiving Party may share some Confidential Information with some of Receiving Party's employees."?

金标：`entailment`；终答标签：`entailment`；初答标签：`contradiction`。

自动选取标志：`revision_wrong_to_correct`。

最终可展示答案：

```text
The contract states that the Recipient may disclose Confidential Information only to its employees, agents, consultants, and contractors on a need-to-know basis, and only if such employees, agents, consultants, and contractors have executed appropriate written agreements with Recipient sufficient to enable Recipient to enforce all the provisions of this Agreement [2, section 2]. This implies that the Recipient can share Confidential Information with some of its employees, but only if they have executed a written agreement, which is consistent with the statement that the Recipient may share some Confidential Information with some of its employees.
```

## 6. llama3 / cuad / Direct

样本：`cuad_test_AMBASSADOREYEWEARGROUPINC_11_17_1997-EX-10.28-ENDORSEMENT AGREEMENT__Parties`。 [完整记录](analysis/CASES.jsonl#L6)

问题：Highlight the parts (if any) of this contract related to "Parties" that should be reviewed by a lawyer. Details: The two or more parties who signed the contract

金标：`present`；终答标签：`invalid`；初答标签：`None`。

自动选取标志：`format_failure`。

最终可展示答案：

```text
（无可展示答案；请结合完整记录中的失败状态或合法空抽取判断）
```

## 7. llama3 / cuad / Direct

样本：`cuad_test_FEDERATEDGOVERNMENTINCOMESECURITIESINC_04_28_2020-EX-99.SERV AGREE-SERVICES AGREEMENT_POWEROF__Expiration Date`。 [完整记录](analysis/CASES.jsonl#L7)

问题：Highlight the parts (if any) of this contract related to "Expiration Date" that should be reviewed by a lawyer. Details: On what date will the contract's initial term expire?

金标：`present`；终答标签：`absent`；初答标签：`None`。

自动选取标志：`gold_covered_but_label_wrong`。

最终可展示答案：

```text
（无可展示答案；请结合完整记录中的失败状态或合法空抽取判断）
```

## 8. llama3 / cuad / RAG

样本：`cuad_test_ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT VENTURE AGREEMENT__Effective Date`。 [完整记录](analysis/CASES.jsonl#L8)

问题：Highlight the parts (if any) of this contract related to "Effective Date" that should be reviewed by a lawyer. Details: The date when the contract is effective 

金标：`present`；终答标签：`present`；初答标签：`None`。

自动选取标志：`out_of_range_citation`。

最终可展示答案：

```text
The Joint Venture shall commence on the 1st of March, 2003,
```

## 9. llama3 / cuad / Agent

样本：`cuad_test_ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT VENTURE AGREEMENT__Termination For Convenience`。 [完整记录](analysis/CASES.jsonl#L9)

问题：Highlight the parts (if any) of this contract related to "Termination For Convenience" that should be reviewed by a lawyer. Details: Can a party terminate this  contract without cause (solely by giving a notice and allowing a waiting  period to expire)?

金标：`absent`；终答标签：`present`；初答标签：`absent`。

自动选取标志：`revision_correct_to_wrong`。

最终可展示答案：

```text
（无可展示答案；请结合完整记录中的失败状态或合法空抽取判断）
```

## 10. llama3 / cuad / Agent

样本：`cuad_test_AFSALABANCORPINC_08_01_1996-EX-1.1-AGENCY AGREEMENT__Agreement Date`。 [完整记录](analysis/CASES.jsonl#L10)

问题：Highlight the parts (if any) of this contract related to "Agreement Date" that should be reviewed by a lawyer. Details: The date of the contract

金标：`present`；终答标签：`present`；初答标签：`absent`。

自动选取标志：`revision_wrong_to_correct`。

最终可展示答案：

```text
Closing Date
```

## 11. mistral / contractnli / Direct

样本：`contractnli_test_117_nda-1`。 [完整记录](analysis/CASES.jsonl#L11)

问题：Is this statement true under the contract: "All Confidential Information shall be expressly identified by the Disclosing Party."?

金标：`contradiction`；终答标签：`neutral`；初答标签：`None`。

自动选取标志：`gold_covered_but_label_wrong`。

最终可展示答案：

```text
The contract does not explicitly state that all Confidential Information must be expressly identified by the Disclosing Party. However, it does define Confidential Information broadly to include any non-public information disclosed for the Business Purpose. Therefore, while the Disclosing Party may choose to label certain information as confidential, it is not a contractual requirement that they do so.
```

## 12. mistral / contractnli / Direct

样本：`contractnli_test_117_nda-16`。 [完整记录](analysis/CASES.jsonl#L12)

问题：Is this statement true under the contract: "Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement."?

金标：`neutral`；终答标签：`invalid`；初答标签：`None`。

自动选取标志：`format_failure`。

最终可展示答案：

```text
（无可展示答案；请结合完整记录中的失败状态或合法空抽取判断）
```

## 13. mistral / contractnli / RAG

样本：`contractnli_test_157_nda-16`。 [完整记录](analysis/CASES.jsonl#L13)

问题：Is this statement true under the contract: "Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement."?

金标：`entailment`；终答标签：`neutral`；初答标签：`None`。

自动选取标志：`out_of_range_citation`。

最终可展示答案：

```text
The contract does not explicitly state that the Receiving Party shall destroy or return all Confidential Information upon the termination of the Agreement. However, it does require the Receiving Party to return all written Confidential Information and delete electronic copies upon receipt of a written request from the Disclosing Party [6]. It is unclear whether this covers all Confidential Information or only written materials. Therefore, the statement is neutral with respect to the contract's text.
```

## 14. mistral / contractnli / Agent

样本：`contractnli_test_181_nda-10`。 [完整记录](analysis/CASES.jsonl#L14)

问题：Is this statement true under the contract: "Receiving Party shall not disclose the fact that Agreement was agreed or negotiated."?

金标：`neutral`；终答标签：`neutral`；初答标签：`contradiction`。

自动选取标志：`revision_wrong_to_correct`。

最终可展示答案：

```text
The contract does not explicitly state that the Receiving Party shall not disclose the fact that the Agreement was agreed or negotiated. [1] However, the contract does require the Receiving Party to keep the Confidential Information of the other party strictly confidential and not make it available to any person other than as permitted [1]. Therefore, it could be inferred that the fact that the Agreement was agreed or negotiated might be considered Confidential Information, but the contract does not explicitly state this.
```

## 15. mistral / contractnli / Agent

样本：`contractnli_test_38_nda-10`。 [完整记录](analysis/CASES.jsonl#L15)

问题：Is this statement true under the contract: "Receiving Party shall not disclose the fact that Agreement was agreed or negotiated."?

金标：`neutral`；终答标签：`invalid`；初答标签：`neutral`。

自动选取标志：`revision_correct_to_wrong`。

最终可展示答案：

```text
（无可展示答案；请结合完整记录中的失败状态或合法空抽取判断）
```

## 16. mistral / cuad / Direct

样本：`cuad_test_ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT VENTURE AGREEMENT__Expiration Date`。 [完整记录](analysis/CASES.jsonl#L16)

问题：Highlight the parts (if any) of this contract related to "Expiration Date" that should be reviewed by a lawyer. Details: On what date will the contract's initial term expire?

金标：`present`；终答标签：`invalid`；初答标签：`None`。

自动选取标志：`format_failure`。

最终可展示答案：

```text
（无可展示答案；请结合完整记录中的失败状态或合法空抽取判断）
```

## 17. mistral / cuad / Direct

样本：`cuad_test_AFSALABANCORPINC_08_01_1996-EX-1.1-AGENCY AGREEMENT__Agreement Date`。 [完整记录](analysis/CASES.jsonl#L17)

问题：Highlight the parts (if any) of this contract related to "Agreement Date" that should be reviewed by a lawyer. Details: The date of the contract

金标：`present`；终答标签：`absent`；初答标签：`None`。

自动选取标志：`gold_covered_but_label_wrong`。

最终可展示答案：

```text
（无可展示答案；请结合完整记录中的失败状态或合法空抽取判断）
```

## 18. mistral / cuad / RAG

样本：`cuad_test_ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT VENTURE AGREEMENT__Agreement Date`。 [完整记录](analysis/CASES.jsonl#L18)

问题：Highlight the parts (if any) of this contract related to "Agreement Date" that should be reviewed by a lawyer. Details: The date of the contract

金标：`absent`；终答标签：`present`；初答标签：`None`。

自动选取标志：`out_of_range_citation`。

最终可展示答案：

```text
The Joint Venture shall commence on the 1st of March, 2003
```

## 19. mistral / cuad / Agent

样本：`cuad_test_ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT VENTURE AGREEMENT__Non-Compete`。 [完整记录](analysis/CASES.jsonl#L19)

问题：Highlight the parts (if any) of this contract related to "Non-Compete" that should be reviewed by a lawyer. Details: Is there a restriction on the ability of a party to compete with the counterparty or operate in a certain geography or business or technology sector? 

金标：`absent`；终答标签：`invalid`；初答标签：`absent`。

自动选取标志：`revision_correct_to_wrong`。

最终可展示答案：

```text
（无可展示答案；请结合完整记录中的失败状态或合法空抽取判断）
```

## 20. mistral / cuad / Agent

样本：`cuad_test_AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agreement__Effective Date`。 [完整记录](analysis/CASES.jsonl#L20)

问题：Highlight the parts (if any) of this contract related to "Effective Date" that should be reviewed by a lawyer. Details: The date when the contract is effective 

金标：`present`；终答标签：`present`；初答标签：`absent`。

自动选取标志：`revision_wrong_to_correct`。

最终可展示答案：

```text
the 31st day of March, 2000
```
