# acl-bea2025-workshop-st
Team LexiLogic's submission to the ACL BEA 2025 Shared Task

https://sig-edu.org/sharedtask/2025

Here are the shared task details, as per the BEA website:

```
Tracks
Track 1 - Mistake Identification: Teams are invited to develop systems to detect whether tutors’ responses recognize mistakes in students’ responses. The following categories are included:
Yes: the mistake is clearly identified/ recognized in the tutor’s response
To some extent: the tutor’s response suggests that there may be a mistake, but it sounds as if the tutor is not certain
No: the tutor does not recognize the mistake (e.g., they proceed to simply provide the answer to the asked question)
Track 2 - Mistake Location: Teams are invited to develop systems to assess whether tutors’ responses accurately point to a genuine mistake and its location in the students’ responses. The following categories are included:
Yes: the tutor clearly points to the exact location of a genuine mistake in the student’s solution
To some extent: the response demonstrates some awareness of the exact mistake, but is vague, unclear, or easy to misunderstand
No: the response does not provide any details related to the mistake
Track 3 - Pedagogical Guidance: Teams are invited to develop systems to evaluate whether tutors’ responses offer correct and relevant guidance, such as an explanation, elaboration, hint, examples, and so on. The following categories are included:
Yes: the tutor provides guidance that is correct and relevant to the student’s mistake
To some extent: guidance is provided but it is fully or partially incorrect, incomplete, or somewhat misleading
No: the tutor’s response does not include any guidance, or the guidance provided is irrelevant to the question or factually incorrect
Track 4 - Actionability: Teams are invited to develop systems to assess whether tutors’ feedback is actionable, i.e., it makes it clear what the student should do next. The following categories are included:
Yes: the response provides clear suggestions on what the student should do next
To some extent: the response indicates that something needs to be done, but it is not clear what exactly that is
No: the response does not suggest any action on the part of the student (e.g., it simply reveals the final answer)
Track 5 - Guess the tutor identity: Teams are invited to develop systems to identify which tutors the anonymized responses in the test set originated from. This track will address 9 classes: expert and novice tutors, and 7 LLMs included in the tutor set.
Evaluation
Tracks 1-4 will use accuracy and macro F1 as the main metrics. These will be used in two settings:
Exact evaluation: predictions submitted by the teams will be evaluated for the exact prediction of the three classes (“Yes”, “To some extent”, and “No”)
Lenient evaluation: since for these dimensions tutor responses annotated as “Yes” and “To some extent” share a certain amount of qualitative value, we will consider “Yes” and “To some extent” as a single class, and evaluate predictions under the 2-class setting (“Yes + To some extent” vs. “No”)
Track 5 will use accuracy of the tutor identity prediction as its main metric.
```
