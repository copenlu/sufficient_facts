# SufficientFacts
This is the dataset SufficientFacts, introduced in the paper "Fact Checking with Insufficient Evidence", accepted at the TACL journal in 2022. 

<p align="center">
  <img src="/data/sufficient_facts/missing.png" width="450" alt="Missing Evidence Example">
</p>


Automating the fact checking (FC) process relies on information obtained from external sources. In this work, we posit that it is crucial for FC models to make veracity predictions only when there is sufficient evidence and otherwise indicate when it is not enough. To this end, we are the first to study what information FC models consider sufficient by introducing a novel task and advancing it with three main contributions. First, we conduct an in-depth empirical analysis of the task with a new fluency-preserving method for omitting information from the evidence at the constituent and sentence level. We identify when models consider the remaining evidence (in)sufficient for FC, based on three trained models with different Transformer architectures and three FC datasets. Second, we ask annotators whether the omitted evidence was important for FC, resulting in a novel diagnostic dataset, **SufficientFacts**, for FC with omitted evidence. We find that models are least successful in detecting missing evidence when adverbial modifiers are omitted (21% accuracy), whereas it is easiest for omitted date modifiers (63% accuracy). Finally, we propose a novel data augmentation strategy for contrastive self-learning of missing evidence by employing the proposed omission method combined with tri-training. It improves performance for Evidence Sufficiency Prediction by up to 17.8 F1 score, which in turn improves FC performance by up to 2.6 F1 score.

## SufficientFacts Description

The dataset consists of three files, each for one of the datasets -- FEVER, HoVer, and VitaminC.
Each file consists of json lines of the format:

```json
{
    "claim": "Unison (Celine Dion album) was originally released by Atlantic Records.", 
    "evidence": [
        [
            "Unison (Celine Dion album)", 
            "The album was originally released on 2 April 1990 ."
        ]
    ],
    "label_before": "REFUTES", 
    "label_after": "NOT ENOUGH", 
    "agreement": "agree_ei", 
    "type": "PP", 
    "removed": ["by Columbia Records"], 
    "text_orig": "[[Unison (Celine Dion album)]] The album was originally released on 2 April 1990 <span style=\"color:red;\">by Columbia Records</span> ."
}
```

Following is each field's description:
* `claim` - the claim that is being verified
* `evidence` - the augmented evidence for the claim, i.e. the evidence with some removed information
* `label_before` - the original label for the claim-evidence pair, before information was removed from the evidence
* `label_after` - the label for the augmented claim-evidence pair, after information was removed from the evidence, as annotated by crowd-source workers
* `type` - type of the information removed from the evidence. The types are fine-grained and their mapping to the general types -- 7 constituent and 1 sentence type can be found in [types.json](types.json) file.
* `removed` - the text of the removed information from the evidence
* `text_orig` - the original text of the evidence, as presented to crowd-source workers, the text of the removed information is inside `<span style=\"color:red;\"></span>` tags.

## SufficientFacts crowd-source annotation

The annotations were performed by workers at Amazon Mechanical Turk. The workers were provided with the following task description:

For each evidence text, some facts have been removed (marked in <span style="color:red;">red</span>). 
You should annotate whether, <b>given the remaining facts in the evidence text, the evidence is still enough for verifying the claim.</b> <br></br>
<ul>
    <li>You should select <i><b>'ENOUGH -- IRRELEVANT'</b></i>, if the <b>remaining information is still <i>enough</i></b> for verifying the claim because the <b>removed information is irrelevant</b> for identifying the evidence as SUPPORTS or REFUTES. See examples 1 and 2.</li>
    <li>You should select <i><b>'ENOUGH -- REPEATED'</b></i>, if the <b>remaining information is still <i>enough</i></b> for verifying the claim because the <b>removed information is relevant but is also present (repeated) in the remaining (not red) text.</b> See example 3.</li>
    <li>You should select <i><b>'NOT ENOUGH'</b></i> -- when <b>1) the removed information is <i>relevant</i></b> for verifying the claim <b> AND 2) it is <i>not present (repeated)</i> in the remaining text.</b> See examples 4, 5, and 6.</li>
    <!--<li>You should select <i><b>'CHANGED INFO'</b></i> in the rare cases when the remaining evidence has <b>changed the support for the claim</b></li>-->
</ul>   

<b>Note: You should not incorporate your own knowledge or beliefs! You should rely only on the evidence provided for the claim.</b> 

The annotators were then given example instance annotations.
Finally, annotators were asked to complete a qualification test in order to be allowed to annotate instances for the task. 
The resulting inter-annotator agreement for SufficientFacts is 0.81 Fleiss'k from three annotators.

