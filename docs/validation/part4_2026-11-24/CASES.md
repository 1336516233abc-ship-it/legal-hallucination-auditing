# 20 Traceable Cases

Selected by deterministic flags for troubleshooting and explanation, not random error rate samples, nor manual hallucination annotation. Each complete record contains full contract text, gold evidence, prompt evidence, and existing internal verification records.

## 1. llama3 / contractnli / Direct

Sample: `contractnli_test_117_nda-8`. [Complete Record](analysis/CASES.jsonl#L1)

Question: Is this statement true under the contract: "Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information."?

Gold: `entailment`; Final answer label: `contradiction`; Initial answer label: `None`。

Automatic selection flag: `gold_covered_but_label_wrong`。

Final displayable answer:

```text
The contract does not contain the statement "Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information."
```

## 2. llama3 / contractnli / RAG

Sample: `contractnli_test_117_nda-15`. [Complete Record](analysis/CASES.jsonl#L2)

Question: Is this statement true under the contract: "Agreement shall not grant Receiving Party any right to Confidential Information."?

Gold: `entailment`; Final answer label: `entailment`; Initial answer label: `None`。

Automatic selection flag: `out_of_range_citation`。

Final displayable answer:

```text
The contract states that "ALL CONFIDENTIAL INFORMATION IS PROVIDED 'AS IS'. THE DISCLOSING PARTY MAKES NO WARRANTIES, EXPRESS, IMPLIED OR OTHERWISE, REGARDING NON-INFRINGEMENT OF THIRD PARTY RIGHTS OR ITS ACCURACY, COMPLETENESS OR PERFORMANCE." [3] This implies that the Disclosing Party does not grant any rights to the Recipient regarding the Confidential Information, including non-infringement of third-party rights.
```

## 3. llama3 / contractnli / RAG

Sample: `contractnli_test_38_nda-15`. [Complete Record](analysis/CASES.jsonl#L3)

Question: Is this statement true under the contract: "Agreement shall not grant Receiving Party any right to Confidential Information."?

Gold: `entailment`; Final answer label: `invalid`; Initial answer label: `None`。

Automatic selection flag: `format_failure`。

Final displayable answer:

```text
(No displayable answer; refer to failure status in complete record or valid empty extraction)
```

## 4. llama3 / contractnli / Agent

Sample: `contractnli_test_117_nda-1`. [Complete Record](analysis/CASES.jsonl#L4)

Question: Is this statement true under the contract: "All Confidential Information shall be expressly identified by the Disclosing Party."?

Gold: `contradiction`; Final answer label: `entailment`; Initial answer label: `contradiction`。

Automatic selection flag: `revision_correct_to_wrong`。

Final displayable answer:

```text
The contract states that the Disclosing Party "may, but is not required to, mark written Confidential Information with the legend 'Confidential' or an equivalent designation" [2, section 1], which implies that the Disclosing Party is not obligated to provide explicit identification of the Confidential Information.
```

## 5. llama3 / contractnli / Agent

Sample: `contractnli_test_117_nda-5`. [Complete Record](analysis/CASES.jsonl#L5)

Question: Is this statement true under the contract: "Receiving Party may share some Confidential Information with some of Receiving Party's employees."?

Gold: `entailment`; Final answer label: `entailment`; Initial answer label: `contradiction`。

Automatic selection flag: `revision_wrong_to_correct`。

Final displayable answer:

```text
The contract states that the Recipient may disclose Confidential Information only to its employees, agents, consultants, and contractors on a need-to-know basis, and only if such employees, agents, consultants, and contractors have executed appropriate written agreements with Recipient sufficient to enable Recipient to enforce all the provisions of this Agreement [2, section 2]. This implies that the Recipient can share Confidential Information with some of its employees, but only if they have executed a written agreement, which is consistent with the statement that the Recipient may share some Confidential Information with some of its employees.
```

## 6. llama3 / cuad / Direct

Sample: `cuad_test_AMBASSADOREYEWEARGROUPINC_11_17_1997-EX-10.28-ENDORSEMENT AGREEMENT__Parties`. [Complete Record](analysis/CASES.jsonl#L6)

Question: Highlight the parts (if any) of this contract related to "Parties" that should be reviewed by a lawyer. Details: The two or more parties who signed the contract

Gold: `present`; Final answer label: `invalid`; Initial answer label: `None`。

Automatic selection flag: `format_failure`。

Final displayable answer:

```text
(No displayable answer; refer to failure status in complete record or valid empty extraction)
```

## 7. llama3 / cuad / Direct

Sample: `cuad_test_FEDERATEDGOVERNMENTINCOMESECURITIESINC_04_28_2020-EX-99.SERV AGREE-SERVICES AGREEMENT_POWEROF__Expiration Date`. [Complete Record](analysis/CASES.jsonl#L7)

Question: Highlight the parts (if any) of this contract related to "Expiration Date" that should be reviewed by a lawyer. Details: On what date will the contract's initial term expire?

Gold: `present`; Final answer label: `absent`; Initial answer label: `None`。

Automatic selection flag: `gold_covered_but_label_wrong`。

Final displayable answer:

```text
(No displayable answer; refer to failure status in complete record or valid empty extraction)
```

## 8. llama3 / cuad / RAG

Sample: `cuad_test_ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT VENTURE AGREEMENT__Effective Date`. [Complete Record](analysis/CASES.jsonl#L8)

Question: Highlight the parts (if any) of this contract related to "Effective Date" that should be reviewed by a lawyer. Details: The date when the contract is effective 

Gold: `present`; Final answer label: `present`; Initial answer label: `None`。

Automatic selection flag: `out_of_range_citation`。

Final displayable answer:

```text
The Joint Venture shall commence on the 1st of March, 2003,
```

## 9. llama3 / cuad / Agent

Sample: `cuad_test_ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT VENTURE AGREEMENT__Termination For Convenience`. [Complete Record](analysis/CASES.jsonl#L9)

Question: Highlight the parts (if any) of this contract related to "Termination For Convenience" that should be reviewed by a lawyer. Details: Can a party terminate this  contract without cause (solely by giving a notice and allowing a waiting  period to expire)?

Gold: `absent`; Final answer label: `present`; Initial answer label: `absent`。

Automatic selection flag: `revision_correct_to_wrong`。

Final displayable answer:

```text
(No displayable answer; refer to failure status in complete record or valid empty extraction)
```

## 10. llama3 / cuad / Agent

Sample: `cuad_test_AFSALABANCORPINC_08_01_1996-EX-1.1-AGENCY AGREEMENT__Agreement Date`. [Complete Record](analysis/CASES.jsonl#L10)

Question: Highlight the parts (if any) of this contract related to "Agreement Date" that should be reviewed by a lawyer. Details: The date of the contract

Gold: `present`; Final answer label: `present`; Initial answer label: `absent`。

Automatic selection flag: `revision_wrong_to_correct`。

Final displayable answer:

```text
Closing Date
```

## 11. mistral / contractnli / Direct

Sample: `contractnli_test_117_nda-1`. [Complete Record](analysis/CASES.jsonl#L11)

Question: Is this statement true under the contract: "All Confidential Information shall be expressly identified by the Disclosing Party."?

Gold: `contradiction`; Final answer label: `neutral`; Initial answer label: `None`。

Automatic selection flag: `gold_covered_but_label_wrong`。

Final displayable answer:

```text
The contract does not explicitly state that all Confidential Information must be expressly identified by the Disclosing Party. However, it does define Confidential Information broadly to include any non-public information disclosed for the Business Purpose. Therefore, while the Disclosing Party may choose to label certain information as confidential, it is not a contractual requirement that they do so.
```

## 12. mistral / contractnli / Direct

Sample: `contractnli_test_117_nda-16`. [Complete Record](analysis/CASES.jsonl#L12)

Question: Is this statement true under the contract: "Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement."?

Gold: `neutral`; Final answer label: `invalid`; Initial answer label: `None`。

Automatic selection flag: `format_failure`。

Final displayable answer:

```text
(No displayable answer; refer to failure status in complete record or valid empty extraction)
```

## 13. mistral / contractnli / RAG

Sample: `contractnli_test_157_nda-16`. [Complete Record](analysis/CASES.jsonl#L13)

Question: Is this statement true under the contract: "Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement."?

Gold: `entailment`; Final answer label: `neutral`; Initial answer label: `None`。

Automatic selection flag: `out_of_range_citation`。

Final displayable answer:

```text
The contract does not explicitly state that the Receiving Party shall destroy or return all Confidential Information upon the termination of the Agreement. However, it does require the Receiving Party to return all written Confidential Information and delete electronic copies upon receipt of a written request from the Disclosing Party [6]. It is unclear whether this covers all Confidential Information or only written materials. Therefore, the statement is neutral with respect to the contract's text.
```

## 14. mistral / contractnli / Agent

Sample: `contractnli_test_181_nda-10`. [Complete Record](analysis/CASES.jsonl#L14)

Question: Is this statement true under the contract: "Receiving Party shall not disclose the fact that Agreement was agreed or negotiated."?

Gold: `neutral`; Final answer label: `neutral`; Initial answer label: `contradiction`。

Automatic selection flag: `revision_wrong_to_correct`。

Final displayable answer:

```text
The contract does not explicitly state that the Receiving Party shall not disclose the fact that the Agreement was agreed or negotiated. [1] However, the contract does require the Receiving Party to keep the Confidential Information of the other party strictly confidential and not make it available to any person other than as permitted [1]. Therefore, it could be inferred that the fact that the Agreement was agreed or negotiated might be considered Confidential Information, but the contract does not explicitly state this.
```

## 15. mistral / contractnli / Agent

Sample: `contractnli_test_38_nda-10`. [Complete Record](analysis/CASES.jsonl#L15)

Question: Is this statement true under the contract: "Receiving Party shall not disclose the fact that Agreement was agreed or negotiated."?

Gold: `neutral`; Final answer label: `invalid`; Initial answer label: `neutral`。

Automatic selection flag: `revision_correct_to_wrong`。

Final displayable answer:

```text
(No displayable answer; refer to failure status in complete record or valid empty extraction)
```

## 16. mistral / cuad / Direct

Sample: `cuad_test_ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT VENTURE AGREEMENT__Expiration Date`. [Complete Record](analysis/CASES.jsonl#L16)

Question: Highlight the parts (if any) of this contract related to "Expiration Date" that should be reviewed by a lawyer. Details: On what date will the contract's initial term expire?

Gold: `present`; Final answer label: `invalid`; Initial answer label: `None`。

Automatic selection flag: `format_failure`。

Final displayable answer:

```text
(No displayable answer; refer to failure status in complete record or valid empty extraction)
```

## 17. mistral / cuad / Direct

Sample: `cuad_test_AFSALABANCORPINC_08_01_1996-EX-1.1-AGENCY AGREEMENT__Agreement Date`. [Complete Record](analysis/CASES.jsonl#L17)

Question: Highlight the parts (if any) of this contract related to "Agreement Date" that should be reviewed by a lawyer. Details: The date of the contract

Gold: `present`; Final answer label: `absent`; Initial answer label: `None`。

Automatic selection flag: `gold_covered_but_label_wrong`。

Final displayable answer:

```text
(No displayable answer; refer to failure status in complete record or valid empty extraction)
```

## 18. mistral / cuad / RAG

Sample: `cuad_test_ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT VENTURE AGREEMENT__Agreement Date`. [Complete Record](analysis/CASES.jsonl#L18)

Question: Highlight the parts (if any) of this contract related to "Agreement Date" that should be reviewed by a lawyer. Details: The date of the contract

Gold: `absent`; Final answer label: `present`; Initial answer label: `None`。

Automatic selection flag: `out_of_range_citation`。

Final displayable answer:

```text
The Joint Venture shall commence on the 1st of March, 2003
```

## 19. mistral / cuad / Agent

Sample: `cuad_test_ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT VENTURE AGREEMENT__Non-Compete`. [Complete Record](analysis/CASES.jsonl#L19)

Question: Highlight the parts (if any) of this contract related to "Non-Compete" that should be reviewed by a lawyer. Details: Is there a restriction on the ability of a party to compete with the counterparty or operate in a certain geography or business or technology sector? 

Gold: `absent`; Final answer label: `invalid`; Initial answer label: `absent`。

Automatic selection flag: `revision_correct_to_wrong`。

Final displayable answer:

```text
(No displayable answer; refer to failure status in complete record or valid empty extraction)
```

## 20. mistral / cuad / Agent

Sample: `cuad_test_AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agreement__Effective Date`. [Complete Record](analysis/CASES.jsonl#L20)

Question: Highlight the parts (if any) of this contract related to "Effective Date" that should be reviewed by a lawyer. Details: The date when the contract is effective 

Gold: `present`; Final answer label: `present`; Initial answer label: `absent`。

Automatic selection flag: `revision_wrong_to_correct`。

Final displayable answer:

```text
the 31st day of March, 2000
```
