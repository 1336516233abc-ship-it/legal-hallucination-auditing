# Part 4 Error and Revision Case List

This file is qualitative case organization of existing experiment outputs, not the course report. Did not re-run models, did not modify original predictions, gold labels, or published metrics.

## Scope and Conclusion Boundaries

- Verified all 60 outputs from two models × three variants, corresponding to 10 same questions (ContractNLI 6, CUAD 4), not 60 independent samples. Below 9 cases selected by different question types, not random sampling, cannot calculate overall error type proportions.
- Original label statistics remain 37/60 correct, 57/60 format pass. Item-by-item separately checked citation ranges, gold evidence coverage, and final display answers; correct labels do not guarantee correct citations or usable answers.
- Among all outputs, 3 format failures, 11 program-recorded grounding_errors. Duplicate samples across variants retained, not counted as independent failure events.
- Among 20 Full outputs, actual revisions 11 times: label wrong→right 1 time, wrong→wrong 4 times, right→right 6 times, right→wrong 0 times. Additionally, 2 revisions introduced new out-of-bounds citations. Only label correction case still has citations pointing to errors; cannot claim 1/11 or 1/5 as independent hallucination revision success rate.
- Qualitative explanations AI-assisted verification of gold and contract original text, not yet undergone dual independent blind review; independent hallucination rate, completeness, abstention precision, and complete revision success rate all remain unevaluated. neutral/absent does not automatically equal abstention.
- Character intervals are Python character indices [start,end) of processed contract context, not PDF page numbers. JSON excerpts all verified equal to original text slices; this page display only removes trailing whitespace. Complete item-by-item index and input SHA256 see [case_analysis.json](case_analysis.json).

## Case Index

|ID|Model / Dataset|Main Observation|
|---|---|---|
|[C01](#c01)|llama3 / contractnli|Label corrected, but citations still point to wrong evidence|
|[C02](#c02)|mistral / contractnli|Explanation already negates claim, label still neutral|
|[C03](#c03)|llama3 / cuad|Re-retrieval found signature date, answer still absent|
|[C04](#c04)|mistral / cuad|Re-retrieval covers renewal notice clause, still misses 180 days|
|[C05](#c05)|mistral / contractnli|Output Label: true, task protocol parsing failed|
|[C06](#c06)|llama3 / cuad|Placeholder passes verbatim match, but doesn't answer target question|
|[C07](#c07)|llama3 / contractnli|Revision introduces out-of-bounds citation, correct label answer cleared|
|[C08](#c08)|mistral / cuad|Label remains correct, revision adds many non-existent citations|
|[C09](#c09)|llama3 / contractnli|Initial citation overflow prevents verification and revision execution|

<a id="c01"></a>

## C01 · Label Corrected, But Citations Still Point to Wrong Evidence

- Sample: `contractnli_test_523_nda-8`
- Question: Is this statement true under the contract: "Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information."?
- Gold: `entailment`; Initial → Final: `contradiction` → `entailment`; Revision executed: Yes.
- Original output: [llama3/contractnli/full Line 4](agent_ablation/llama3/contractnli/full/predictions.jsonl#L4). 

**Initial Model Response**

```text
Label: contradiction
Answer: The contract does not require the Receiving Party to notify the Disclosing Party in case the Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information. Instead, it states that the Receiving Party will "immediately notify the Company of the existence, terms and circumstances surrounding such request" and "consult with the Company on the advisability of taking legally available steps to resist or narrow such request" (Evidence [2]).
```

**Final Model Response**

```text
Label: entailment
Answer: The contract requires the Receiving Party to "immediately notify the Company of the existence, terms and circumstances surrounding such request" and "consult with the Company on the advisability of taking legally available steps to resist or narrow such request" if the Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information (Evidence [2]).
```

Final display: Model answer retained.

|Stage|Input Evidence Chunks|Gold Interval Coverage|Model Citation Numbers|
|---|---:|---|---|
|Initial|7|1.0|[2]|
|Final|3|1.0|[2]|

**Contract Original Text and Verification**

Dataset Gold Evidence, `[4181,4495)`：

```text
In the event that you or any of your representatives received a request or are required (by deposition, interrogatory, request for documents, subpoena, civil investigative demand or similar process) to disclose all or any part of the Evaluation Materials, you or your representatives, as the case may be, agree to
```

Dataset Gold Evidence, `[4495,4598)`：

```text
(i) immediately notify the Company of the existence, terms and circumstances surrounding such request,
```

Initial contradiction conflicts with the notification obligation cited by the model itself; after revision entailment consistent with gold and contract. but re-retrieval moved notification obligation from evidence [2] to [1], Finalstill writes Evidence [2]. can only call label correction, cannot claim complete evidence citation fix. number retention is observation result, internal reason not verified through new experiments. 

**Variant Comparison for Same Sample**

|Variant|Final Label|Label Correct|Format Pass|Citation/Extraction Check Error|
|---|---|---|---|---|
|full|entailment|Yes|Yes|No program flag; does not represent semantic verification|
|no_verification|contradiction|No|Yes|No program flag; does not represent semantic verification|
|no_revision|contradiction|No|Yes|No program flag; does not represent semantic verification|

Follow-up check direction: Check whether post-revision citations support corresponding claims, not just whether numbers are out of bounds. 


<a id="c02"></a>

## C02 · Explanation Already Negates Claim, Label Still Neutral

- Sample: `contractnli_test_493_nda-2`
- Question: Is this statement true under the contract: "Confidential Information shall only include technical information."?
- Gold: `contradiction`; Initial → Final: `neutral` → `neutral`; Revision executed: Yes.
- Original output: [mistral/contractnli/full Line 2](agent_ablation/mistral/contractnli/full/predictions.jsonl#L2). 

**Initial Model Response**

```text
Label: neutral
Answer: The contract does not explicitly state that Confidential Information shall only include technical information. [1] However, the definition of Confidential Information in the contract includes various types of information such as trade secrets, know-how, inventions, designs, methods, techniques, systems, processes, software programs, works of authorship, business plans, customer lists, projects, plans, pricing, proposals, and any other information. [2] Therefore, while technical information is included in the definition, it is not the only type of Confidential Information.
```

**Final Model Response**

```text
Label: neutral
Answer: The contract does not state that Confidential Information only includes technical information [1]. While technical information is included in the definition of Confidential Information, other types of information are also covered [1].
```

Final display: Model answer retained.

|Stage|Input Evidence Chunks|Gold Interval Coverage|Model Citation Numbers|
|---|---:|---|---|
|Initial|4|1.0|[1, 2]|
|Final|3|1.0|[1, 1]|

**Contract Original Text and Verification**

Dataset Gold Evidence, `[936,1276)`：

```text
A. "Confidential Information" of a party shall mean any trade secrets, know-how, inventions, products, designs, methods, techniques, systems, processes, software programs, works of authorship, business plans, customer lists, projects, plans, pricing, proposals and any other information which a party discloses to the Recipient Party that:
```

question contains only technical information. contract definition explicitly lists business plans, customer lists, pricing etc scope, gold is contradiction. Initialexplanation and revision explanation both acknowledge more than technical information, but label always neutral; this is not missing gold evidence, but label-explanation inconsistency. 

**Variant Comparison for Same Sample**

|Variant|Final Label|Label Correct|Format Pass|Citation/Extraction Check Error|
|---|---|---|---|---|
|full|neutral|No|Yes|No program flag; does not represent semantic verification|
|no_verification|neutral|No|Yes|No program flag; does not represent semantic verification|
|no_revision|neutral|No|Yes|No program flag; does not represent semantic verification|

Follow-up check direction: Verify consistency of qualifiers like only, all and label-explanation consistency. 


<a id="c03"></a>

## C03 · Re-retrieval Found Signature Date, Answer Still Absent

- Sample: `cuad_test_HPILHOLDING_01_07_2015-EX-99.1-COOPERATION AGREEMENT__Agreement Date`
- Question: Highlight the parts (if any) of this contract related to "Agreement Date" that should be reviewed by a lawyer. Details: The date of the contract
- Gold: `present`; Initial → Final: `absent` → `absent`; Revision executed: Yes.
- Original output: [llama3/cuad/full Line 2](agent_ablation/llama3/cuad/full/predictions.jsonl#L2). 

**Initial Model Response**

```text
{"label":"absent","extracted_spans":[],"citations":[]}
```

**Final Model Response**

```text
{"label":"absent","extracted_spans":[],"citations":[]}
```

Final display: Valid absent empty extraction.

|Stage|Input Evidence Chunks|Gold Interval Coverage|Model Citation Numbers|
|---|---:|---|---|
|Initial|3|0.0|[]|
|Final|3|1.0|[]|

**Contract Original Text and Verification**

Dataset Gold Evidence, `[3560,3584)`：

```text
5th day of January, 2015
```

signature date context, `[3504,3605)`：

```text
THIS COOPERATION AGREEMENT ("Agreement") is signed this 5th day of January, 2015 (the "Closing Date")
```

Initialprompt did not cover date gold; after re-retrieval evidence [3] covers signature date, label still absent. this sample simultaneously showsInitialretrieval omission and generation omission after re-retrieval, cannot attribute only to retrieval. Mistral 's threeVariantall extracted this date, can serve as comparison for same question. 

**Variant Comparison for Same Sample**

|Variant|Final Label|Label Correct|Format Pass|Citation/Extraction Check Error|
|---|---|---|---|---|
|full|absent|No|Yes|No program flag; does not represent semantic verification|
|no_verification|absent|No|Yes|No program flag; does not represent semantic verification|
|no_revision|absent|No|Yes|No program flag; does not represent semantic verification|

Follow-up check direction: separately checkInitialand target evidence in revision prompt, then evaluate whether model uses it. 


<a id="c04"></a>

## C04 · Re-retrieval Covers Renewal Notice Clause, Still Misses 180 Days

- Sample: `cuad_test_PACIRA PHARMACEUTICALS, INC. - A_R STRATEGIC LICENSING, DISTRIBUTION AND MARKETING AGREEMENT __Notice Period To Terminate Renewal`
- Question: Highlight the parts (if any) of this contract related to "Notice Period To Terminate Renewal" that should be reviewed by a lawyer. Details: What is the notice period required to terminate renewal?
- Gold: `present`; Initial → Final: `absent` → `absent`; Revision executed: Yes.
- Original output: [mistral/cuad/full Line 4](agent_ablation/mistral/cuad/full/predictions.jsonl#L4). 

**Initial Model Response**

```text
{"label":"absent","extracted_spans":[],"citations":[]}
```

**Final Model Response**

```text
Based on the provided evidence, there is no explicit clause stating the notice period to terminate a renewal. The contract does mention a notice period for termination by either party under certain circumstances (Section 16.1), but it does not specify a notice period for terminating a renewal term. Therefore, the label should be "absent" and the answer should be:

{"label":"absent","extracted_spans":[],"citations":[]}
```

Final display: Valid absent empty extraction.

|Stage|Input Evidence Chunks|Gold Interval Coverage|Model Citation Numbers|
|---|---:|---|---|
|Initial|5|0.0|[]|
|Final|3|1.0|[]|

**Contract Original Text and Verification**

Dataset Gold Evidence, `[94251,94488)`：

```text
Notwithstanding the foregoing, this Agreement can be terminated by EKR at the end of the Initial Term by delivery of written notice to PPI at least one hundred eighty (180) days prior to the end of the Initial Term or any renewal term.
```

Initialprompt did not cover gold, after re-retrieval evidence [2] already contains at least one hundred eighty (180) days and any renewal term. Finalstill states no explicit notice period to terminate renewal, predicted absent, with gold present inconsistent. 

**Variant Comparison for Same Sample**

|Variant|Final Label|Label Correct|Format Pass|Citation/Extraction Check Error|
|---|---|---|---|---|
|full|absent|No|Yes|No program flag; does not represent semantic verification|
|no_verification|absent|No|Yes|No program flag; does not represent semantic verification|
|no_revision|absent|No|Yes|No program flag; does not represent semantic verification|

Follow-up check direction: Review clauses combining complete conditions and renewal term, avoid misjudging existing renewal conditions as absent. 


<a id="c05"></a>

## C05 · Output Label: true, Task Protocol Parsing Failed

- Sample: `contractnli_test_523_nda-8`
- Question: Is this statement true under the contract: "Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information."?
- Gold: `entailment`; Initial → Final: `invalid` → `invalid`; Revision executed: No.
- Original output: [mistral/contractnli/full Line 4](agent_ablation/mistral/contractnli/full/predictions.jsonl#L4). 

**Initial Model Response**

```text
Label: true
Answer: The contract requires the Receiving Party to notify the Disclosing Party if they are required by law, regulation, or judicial process to disclose any Confidential Information [1].
```

**Final Model Response**

```text
Label: true
Answer: The contract requires the Receiving Party to notify the Disclosing Party if they are required by law, regulation, or judicial process to disclose any Confidential Information [1].
```

Final display: Cleared due to parsing or citation check error.

|Stage|Input Evidence Chunks|Gold Interval Coverage|Model Citation Numbers|
|---|---:|---|---|
|Initial|7|1.0|[]|
|Final|7|1.0|[]|

**Contract Original Text and Verification**

Dataset Gold Evidence, `[4181,4495)`：

```text
In the event that you or any of your representatives received a request or are required (by deposition, interrogatory, request for documents, subpoena, civil investigative demand or similar process) to disclose all or any part of the Evaluation Materials, you or your representatives, as the case may be, agree to
```

Dataset Gold Evidence, `[4495,4598)`：

```text
(i) immediately notify the Company of the existence, terms and circumstances surrounding such request,
```

Initialoriginal response uses Label: true, while ContractNLI only accepts entailment, contradiction, neutral. threeVariantall returned same invalid label and counted as error; Full skipped verification and revision due to parsing failure, Finaldisplay answer empty. even if explanation text mentions notification obligation, cannot change record toFormat PassorLabel Correct. 

**Variant Comparison for Same Sample**

|Variant|Final Label|Label Correct|Format Pass|Citation/Extraction Check Error|
|---|---|---|---|---|
|full|invalid|No|No|No program flag; does not represent semantic verification|
|no_verification|invalid|No|No|No program flag; does not represent semantic verification|
|no_revision|invalid|No|No|No program flag; does not represent semantic verification|

Follow-up check direction: subsequently separately verify structured output or format recovery strategy; cannot retroactively modify original predictions when summarizing results. 


<a id="c06"></a>

## C06 · Placeholder Passes Verbatim Match, But Doesn't Answer Target Question

- Sample: `cuad_test_N2KINC_10_16_1997-EX-10.16-SPONSORSHIP AGREEMENT__Notice Period To Terminate Renewal`
- Question: Highlight the parts (if any) of this contract related to "Notice Period To Terminate Renewal" that should be reviewed by a lawyer. Details: What is the notice period required to terminate renewal?
- Gold: `absent`; Initial → Final: `present` → `present`; Revision executed: No.
- Original output: [llama3/cuad/full Line 3](agent_ablation/llama3/cuad/full/predictions.jsonl#L3). 

**Initial Model Response**

```text
{"label":"present","extracted_spans":["[****]"],"citations":[2]}
```

**Final Model Response**

```text
{"label":"present","extracted_spans":["[****]"],"citations":[2]}
```

Final display: Model answer retained.

|Stage|Input Evidence Chunks|Gold Interval Coverage|Model Citation Numbers|
|---|---:|---|---|
|Initial|5|no gold interval, not counted|[2]|
|Final|5|no gold interval, not counted|[2]|

**Contract Original Text and Verification**

renewal negotiation context; non-gold notice period, `[6380,7203)`：

```text
ii) Commencing not later than [****] prior to the expiration of the term of the Agreement, Excite will negotiate with Sponsor in good faith with respect to the terms and conditions under which this Agreement would be renewed. Excite will negotiate exclusively with Sponsor for the next [****] in good faith effort to negotiate and execute a written sponsorship renewal agreement. If, [****] prior to the expiration of the term of the Agreement, the parties have not entered into a written sponsorship renewal agreement. Excite may enter into negotiations with any third party with respect to retail music store sponsorships of the Excite Site.
```

gold is absent. but model treats redacted placeholder [****] as present extracted answer. this string indeed appears in input, so verbatim match passes; Verifier 's supported reason cites advertising impressions, does not prove termination renewal notice period. Full no revision because detection not triggered. here can confirm local task relevance checking insufficient, cannot estimate overall hallucination rate from this. 

**Variant Comparison for Same Sample**

|Variant|Final Label|Label Correct|Format Pass|Citation/Extraction Check Error|
|---|---|---|---|---|
|full|present|No|Yes|No program flag; does not represent semantic verification|
|no_verification|present|No|Yes|No program flag; does not represent semantic verification|
|no_revision|present|No|Yes|No program flag; does not represent semantic verification|

Follow-up check direction: Verifier should check whether extracted content answers target clause question, not just whether string appears. 


<a id="c07"></a>

## C07 · Revision Introduces Out-of-Bounds Citation, Correct Label Answer Cleared

- Sample: `contractnli_test_53_nda-18`
- Question: Is this statement true under the contract: "Receiving Party shall not solicit some of Disclosing Party's representatives."?
- Gold: `neutral`; Initial → Final: `neutral` → `neutral`; Revision executed: Yes.
- Original output: [llama3/contractnli/full Line 5](agent_ablation/llama3/contractnli/full/predictions.jsonl#L5). 

**Initial Model Response**

```text
Label: neutral
Answer: The document does not specify that the Receiving Party shall not solicit some of Disclosing Party's representatives.
```

**Final Model Response**

```text
Label: neutral
Answer: The document does not specify that the Receiving Party shall not solicit some of Disclosing Party's representatives.

The original question asks if the Receiving Party shall not solicit some of Disclosing Party's representatives. However, the contract does not explicitly state this. The closest relevant clause is clause 18, which states that "Nothing in this Agreement is intended to or shall operate to create a partnership or joint venture of any kind between the Parties (or any of them), or to authorise any Party to act as agent for another, and no Party shall have authority to act in the name or on behalf of or otherwise to bind any other in any way." [18] This clause does not specifically prohibit the Receiving Party from soliciting representatives of the Disclosing Party. Therefore, the answer is neutral, as the document does not provide sufficient information to determine whether the Receiving Party shall not solicit some of Disclosing Party's representatives.
```

Final display: Cleared due to parsing or citation check error.

|Stage|Input Evidence Chunks|Gold Interval Coverage|Model Citation Numbers|
|---|---:|---|---|
|Initial|6|no gold interval, not counted|[]|
|Final|3|no gold interval, not counted|[18]|

**Contract Original Text and Verification**

contractLine 18 clause; used to explain numbering confusion, not solicitation restriction evidence, `[13542,13861)`：

```text
18. Nothing in this Agreement is intended to or shall operate to create a partnership or joint venture of any kind between the Parties (or any of them), or to authorise any Party to act as agent for another, and no Party shall have authority to act in the name or on behalf of or otherwise to bind any other in any way.
```

InitialandFinallabels all neutral, consistent with gold. revision response added [18], butFinalinput only has 3 evidence chunks. original contract indeed has clause 18, content involves non-partnership and agency relationship; it is neither evidence number [18], nor solicitation prohibition clause. range check accordingly marks grounding_error, display answer empty. this is new citation error, cannot call revision successful. neutral itself not used as manual abstention label. 

**Variant Comparison for Same Sample**

|Variant|Final Label|Label Correct|Format Pass|Citation/Extraction Check Error|
|---|---|---|---|---|
|full|neutral|Yes|Yes|answer cited evidence outside the model input|
|no_verification|neutral|Yes|Yes|No program flag; does not represent semantic verification|
|no_revision|neutral|Yes|Yes|No program flag; does not represent semantic verification|

Follow-up check direction: Separate contract clause numbers from prompt evidence numbers, and check whether revision introduces new errors. 


<a id="c08"></a>

## C08 · Label Remains Correct, Revision Adds Many Non-existent Citations

- Sample: `cuad_test_ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT VENTURE AGREEMENT__Notice Period To Terminate Renewal`
- Question: Highlight the parts (if any) of this contract related to "Notice Period To Terminate Renewal" that should be reviewed by a lawyer. Details: What is the notice period required to terminate renewal?
- Gold: `absent`; Initial → Final: `absent` → `absent`; Revision executed: Yes.
- Original output: [mistral/cuad/full Line 1](agent_ablation/mistral/cuad/full/predictions.jsonl#L1). 

**Initial Model Response**

```text
{"label":"absent","extracted_spans":[],"citations":[]}
```

**Final Model Response**

```text
{"label":"absent","extracted_spans":[],"citations":[1,2,3,4,5,11,12,13,14,15,16]}
```

Final display: Cleared due to parsing or citation check error.

|Stage|Input Evidence Chunks|Gold Interval Coverage|Model Citation Numbers|
|---|---:|---|---|
|Initial|5|no gold interval, not counted|[]|
|Final|3|no gold interval, not counted|[1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16]|

**Contract Original Text and Verification**

joint venture term and extension agreement; gold absent has no target evidence interval, `[5991,6263)`：

```text
8. Term of Joint Venture. The Joint Venture shall commence on the 1st of March, 2003, and shall be effective until February 28, 2004 unless extended by written agreement of the Joint Venturers not less than thirty (30) days prior to scheduled termination.
```

Initial absent consistent with gold and no citation errors. after revision still absent, but added [1,2,3,4,5,11,12,13,14,15,16]; Finalonly has 3 evidence chunks, 8 numbers out of bounds. absent empty extraction originally valid structure, cannot judge wrong only because answer blank; here the exact new problem is citation overflow. 

**Variant Comparison for Same Sample**

|Variant|Final Label|Label Correct|Format Pass|Citation/Extraction Check Error|
|---|---|---|---|---|
|full|absent|Yes|Yes|answer cited evidence outside the model input|
|no_verification|absent|Yes|Yes|No program flag; does not represent semantic verification|
|no_revision|absent|Yes|Yes|No program flag; does not represent semantic verification|

Follow-up check direction: Separately count label changes, valid empty extractions, and new citation errors, avoid mixing them into one success rate. 


<a id="c09"></a>

## C09 · Initial Citation Overflow Prevents Verification and Revision Execution

- Sample: `contractnli_test_4_nda-4`
- Question: Is this statement true under the contract: "Receiving Party shall not use any Confidential Information for any purpose other than the purposes stated in Agreement."?
- Gold: `entailment`; Initial → Final: `entailment` → `entailment`; Revision executed: No.
- Original output: [llama3/contractnli/full Line 3](agent_ablation/llama3/contractnli/full/predictions.jsonl#L3). 

**Initial Model Response**

```text
Label: entailment
Answer: The Recipient undertakes not to use the Confidential Information for any purpose except the Purpose, without first obtaining the written agreement of the Discloser. [2]
```

**Final Model Response**

```text
Label: entailment
Answer: The Recipient undertakes not to use the Confidential Information for any purpose except the Purpose, without first obtaining the written agreement of the Discloser. [2]
```

Final display: Cleared due to parsing or citation check error.

|Stage|Input Evidence Chunks|Gold Interval Coverage|Model Citation Numbers|
|---|---:|---|---|
|Initial|1|1.0|[2]|
|Final|1|1.0|[2]|

**Contract Original Text and Verification**

Dataset Gold Evidence, `[444,611)`：

```text
2. The Recipient undertakes not to use the Confidential Information for any purpose except the Purpose, without first obtaining the written agreement of the Discloser.
```

contractLine 2 clause supports usage restriction, predicted entailment also correct. butmodel uses [2], while input only has 1 evidence chunks; grounding_error causes verification and revision to be skipped, Final display empty. Unlike C07, this error atInitialstage already appeared, cannot attribute to reviser. 

**Variant Comparison for Same Sample**

|Variant|Final Label|Label Correct|Format Pass|Citation/Extraction Check Error|
|---|---|---|---|---|
|full|entailment|Yes|Yes|answer cited evidence outside the model input|
|no_verification|entailment|Yes|Yes|answer cited evidence outside the model input|
|no_revision|entailment|Yes|Yes|answer cited evidence outside the model input|

Follow-up check direction: Prioritize checking mixed use of evidence numbers and contract clause numbers; then separately evaluate whether format or citation recovery branch needed. 

## All Full Output Index

Retains correct, incorrect, and format failure outputs for convenient case selection scope verification.

|Model / Dataset|Sample (can open complete trace)|Gold|Initial → Final|Revision|grounding_error|
|---|---|---|---|---|---|
|llama3 / contractnli|[41_nda-1](agent_ablation/llama3/contractnli/full/predictions.jsonl#L1)|contradiction|contradiction → contradiction|No|No program flag|
|llama3 / contractnli|[493_nda-2](agent_ablation/llama3/contractnli/full/predictions.jsonl#L2)|contradiction|contradiction → contradiction|No|No program flag|
|llama3 / contractnli|[4_nda-4](agent_ablation/llama3/contractnli/full/predictions.jsonl#L3)|entailment|entailment → entailment|No|Yes|
|llama3 / contractnli|[523_nda-8](agent_ablation/llama3/contractnli/full/predictions.jsonl#L4)|entailment|contradiction → entailment|Yes|No program flag|
|llama3 / contractnli|[53_nda-18](agent_ablation/llama3/contractnli/full/predictions.jsonl#L5)|neutral|neutral → neutral|Yes|Yes|
|llama3 / contractnli|[59_nda-13](agent_ablation/llama3/contractnli/full/predictions.jsonl#L6)|neutral|neutral → neutral|Yes|No program flag|
|llama3 / cuad|[ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT VENTURE AGREEMENT__Notice Period To Terminate Renewal](agent_ablation/llama3/cuad/full/predictions.jsonl#L1)|absent|present → present|No|Yes|
|llama3 / cuad|[HPILHOLDING_01_07_2015-EX-99.1-COOPERATION AGREEMENT__Agreement Date](agent_ablation/llama3/cuad/full/predictions.jsonl#L2)|present|absent → absent|Yes|No program flag|
|llama3 / cuad|[N2KINC_10_16_1997-EX-10.16-SPONSORSHIP AGREEMENT__Notice Period To Terminate Renewal](agent_ablation/llama3/cuad/full/predictions.jsonl#L3)|absent|present → present|No|No program flag|
|llama3 / cuad|[PACIRA PHARMACEUTICALS, INC. - A_R STRATEGIC LICENSING, DISTRIBUTION AND MARKETING AGREEMENT __Notice Period To Terminate Renewal](agent_ablation/llama3/cuad/full/predictions.jsonl#L4)|present|present → present|No|Yes|
|mistral / contractnli|[41_nda-1](agent_ablation/mistral/contractnli/full/predictions.jsonl#L1)|contradiction|neutral → neutral|Yes|No program flag|
|mistral / contractnli|[493_nda-2](agent_ablation/mistral/contractnli/full/predictions.jsonl#L2)|contradiction|neutral → neutral|Yes|No program flag|
|mistral / contractnli|[4_nda-4](agent_ablation/mistral/contractnli/full/predictions.jsonl#L3)|entailment|entailment → entailment|No|No program flag|
|mistral / contractnli|[523_nda-8](agent_ablation/mistral/contractnli/full/predictions.jsonl#L4)|entailment|invalid → invalid|No|No program flag|
|mistral / contractnli|[53_nda-18](agent_ablation/mistral/contractnli/full/predictions.jsonl#L5)|neutral|neutral → neutral|Yes|No program flag|
|mistral / contractnli|[59_nda-13](agent_ablation/mistral/contractnli/full/predictions.jsonl#L6)|neutral|neutral → neutral|Yes|No program flag|
|mistral / cuad|[ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT VENTURE AGREEMENT__Notice Period To Terminate Renewal](agent_ablation/mistral/cuad/full/predictions.jsonl#L1)|absent|absent → absent|Yes|Yes|
|mistral / cuad|[HPILHOLDING_01_07_2015-EX-99.1-COOPERATION AGREEMENT__Agreement Date](agent_ablation/mistral/cuad/full/predictions.jsonl#L2)|present|present → present|No|No program flag|
|mistral / cuad|[N2KINC_10_16_1997-EX-10.16-SPONSORSHIP AGREEMENT__Notice Period To Terminate Renewal](agent_ablation/mistral/cuad/full/predictions.jsonl#L3)|absent|absent → absent|Yes|No program flag|
|mistral / cuad|[PACIRA PHARMACEUTICALS, INC. - A_R STRATEGIC LICENSING, DISTRIBUTION AND MARKETING AGREEMENT __Notice Period To Terminate Renewal](agent_ablation/mistral/cuad/full/predictions.jsonl#L4)|present|absent → absent|Yes|No program flag|

## Follow-up Priorities

1. Check citation numbers: cover both out-of-bounds and valid numbers pointing to wrong content; numbers after re-retrieval need re-mapping.
2. Check task relevance: verbatim extraction or verifier supported flag cannot substitute for answering target question. 
3. Check label and explanation consistency, and missed answers after re-retrieval obtains target clause.
4. Separately verify format recovery strategy, preserve original failure records, cannot retroactively rewrite published results.
Above are check directions proposed by cases; this iteration did not change Agent implementation nor run post-fix control experiments. 

## Reproduction and Completeness

Execute in repository root (standard library only, does not load models):

```bash
python3 docs/validation/part4_2026-11-22/build_case_analysis.py
```

Requires original processed test files and 12 JSONL files in this directory.Script verifies fixed samples between models, data SHA256, all 60 Final scores, Initial/Final evidence slices, citation ranges and display suppression conditions; simultaneously verifies label changes and evidence positions specific to above cases. 
Original experiment main table:[RESULTS.md](RESULTS.md). Independent review protocol:[ANSWER_SUPPORT_PROTOCOL.md](../../ANSWER_SUPPORT_PROTOCOL.md). 
