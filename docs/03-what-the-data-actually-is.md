# What the data actually is, and the one decision it forces

This document assumes nothing. It explains what the Open University recorded,
what we are trying to build from it, and the single design decision that
determines whether the result is useful or worthless.

No machine learning knowledge required. Terms are defined as they appear.

---

## 1. The setting

The Open University is a distance-learning university in the UK. Students do not
attend campus — the course lives on a website. They read pages, post in forums,
watch material, submit assignments, all online.

Which means the university has a **complete log of everything every student
clicked**. That is unusual, and it is why this dataset exists.

Some vocabulary from the data:

- A **module** is a course. They have codes: `AAA`, `BBB`, `CCC` … `GGG`. Seven
  of them. The real subject names are withheld for anonymity.
- A **presentation** is one particular run of that course, like a term. They are
  coded by year and month: `2013B` started in February 2013, `2013J` in October
  2013. (`B` and `J` are the OU's own letters for February and October starts.)
- So `DDD_2014J` means "module DDD, the run that started October 2014".

The dataset covers 22 module-presentations, 32,593 students, 2013–2014.

**One row of our analysis = one student on one course in one term.** The word
for that is an **enrolment**. A single person can appear several times — they
might take BBB in 2013 and CCC in 2014, or retake the same module. That detail
matters later.

---

## 2. What was recorded

Seven files. Here is what each one is, in plain terms.

### `studentInfo` — who they are and how it ended

One row per enrolment. Contains demographics (gender, age band, region, prior
education, a deprivation measure for their home area, whether they declared a
disability, how many credits they were studying, how many times they had
attempted this module before).

And critically, one column called `final_result`, which is one of four values:

| Value | Meaning |
|---|---|
| `Pass` | Completed and passed |
| `Distinction` | Completed and passed well |
| `Fail` | Stayed enrolled to the end, did not pass |
| `Withdrawn` | Left the course before the end |

The actual counts, out of 32,593 enrolments:

| Outcome | Count | Share |
|---|---:|---:|
| Pass | 12,361 | 37.9% |
| Withdrawn | 10,156 | 31.2% |
| Fail | 7,052 | 21.6% |
| Distinction | 3,024 | 9.3% |

So **52.8% of enrolments end in Fail or Withdrawn** — slightly more than half.

That number is worth pausing on, because it is higher than people expect and it
changes how the problem should be described. This is not a rare-event problem.
The thing we are trying to detect is the *majority* outcome. Distance learning
is hard, and most enrolments do not end in a pass.

One practical consequence: "flag the at-risk students" cannot mean flagging
everyone the model thinks is at risk, because that would be half the cohort and
no tutor has that much time. It has to mean *ranking* students and working down
the list. That distinction shapes how we measure success later.

### `studentVle` — the clickstream

The big one: about 10.7 million rows.

One row per **student, per webpage, per day**, recording how many times they
clicked it. So if student 11391 clicked the course homepage 4 times on a given
day, that is one row saying so.

This is the behavioural record. It is where the interesting signal lives.

### `vle` — what each webpage is

`studentVle` only gives you a numeric page ID. This file translates it into a
type: was it a forum, a quiz, a content page, a resource link? Twenty-ish
categories. You join this on to make clicks interpretable — "spent time in
forums" means something, "clicked page 546652" does not.

### `studentAssessment` and `assessments`

`assessments` lists each assignment: its type (tutor-marked, computer-marked, or
the final exam), its due date, and how much it counts toward the final mark.

`studentAssessment` records submissions: which student submitted which
assessment, on what day, and what they scored.

One subtlety worth noticing: **only submissions appear here.** A student who
never handed anything in has no rows at all. So a missing row is itself
informative — it means "did not submit", not "we don't know".

### `studentRegistration` — the dates that matter

One row per enrolment, with two columns:

- `date_registration` — when they signed up
- `date_unregistration` — when they dropped out, **or empty if they never did**

That second column is the centre of this whole document.

### `courses`

Trivial: how many days long each module-presentation was. Around 234–269 days.

---

## 3. The date convention — read this twice

**There are no calendar dates anywhere in this dataset.** Every date is an
integer counting days from the start of that presentation.

- Day `0` = the first day of teaching
- Day `28` = four weeks in
- Day `-30` = a month *before* teaching started

So `date_registration = -52` means "registered 52 days before the course began",
which is completely normal. And `date_unregistration = 14` means "dropped out
two weeks into the course".

Negative numbers are not errors. They are the pre-course period, and quite a lot
happens there.

---

## 4. What we are building

An **early-warning system**.

Concretely: four weeks into a course, feed the system everything known about
each student so far. It returns a list of students ranked by how likely they are
to fail or withdraw. A tutor takes the top twenty and reaches out.

That is the whole product. Everything else is in service of it.

Two things follow immediately, and they are not obvious:

**It only counts if it is early.** A prediction made in week 20 that a student
will fail is correct and useless — there is no time left to do anything. So we
are not trying to maximise accuracy. We are trying to find the earliest point at
which the prediction is *good enough to act on*. Those are different goals and
they pull in opposite directions.

**It is a ranking, not a verdict.** Nobody intervenes on 32,593 students. The
question is never "is this student at risk, yes or no" — it is "who are the
twenty most worth my time this week". This changes how we measure success, in
ways covered in the glossary.

---

## 5. The decision

Now the part that actually needed explaining.

### The setup

We have to pick a moment to make the prediction. Call it the **cutoff** — say
day 28, four weeks in. The rule is that the system may only look at things that
happened on or before day 28. Anything later has not happened yet in real life,
so using it would be cheating.

Fine. But now look at `date_unregistration` again.

Students drop out **throughout** the course, and a lot of them do it early. Some
drop out before day 0 — they registered, then changed their mind before teaching
even started. In the real data, a large share of all withdrawals happen in the
first few weeks.

So on day 28, when we run the system, the students split into two groups:

- **Group A — still enrolled.** Nobody knows how they will end up. This is a
  genuine open question.
- **Group B — already dropped out.** They unregistered on day 6, day 12, day 20.
  The university has already processed it. Their outcome is a settled fact,
  recorded in the database.

**The decision is: does the system make predictions about Group B?**

### The instinct is "yes, of course"

More data is better, surely? Every instinct in you says do not throw away rows.

That instinct is wrong here, in two ways — one obvious, one that quietly
destroys the project.

### Problem one: it is not a prediction

You cannot predict something that already happened and was written down. If the
university wants to know who has withdrawn, it looks it up. Instantly. For free.

Including Group B means padding your results with cases that were never
uncertain. Whatever number you report at the end is then partly measuring your
ability to look things up.

### Problem two: it corrupts what the system learns

This is the one that matters.

Think about what Group B looks like in the click data. A student who withdrew on
day 10 **stopped clicking on day 10**. They have to — they are gone. So from day
11 to day 28, their record is silence. Nothing. Eighteen consecutive empty days.

Now, the way these systems work is that a computer searches the data for
patterns that separate the two outcomes. It does not know what any column means;
it just looks for whatever splits the groups most cleanly.

So what does it find? It finds:

> **"No clicks in the last two weeks" → will not pass.**

And that rule is *right essentially every time* — for Group B. Of course it is.
It is not detecting risk. It is detecting absence, in people who are absent
because they already left.

Two things now go wrong at once.

**The score looks superb.** Whatever number measures performance shoots up,
because a large chunk of your cases are now trivially easy. You would report an
excellent result and believe the project succeeded.

**The system stops looking for anything else.** These methods concentrate effort
wherever the biggest gains are. The already-withdrawn students are a large,
easy, high-payoff group, so the pattern-finding piles into them. The subtle
signals — a still-enrolled student whose forum activity has halved, who is
handing things in three days later than they used to, who has stopped opening
the reading — are comparatively small potatoes. They get ignored.

Which is a catastrophe, because **Group A is the entire point.** They are the
only students a tutor can still do anything about. You have built a system that
is excellent at the one job nobody needs and poor at the only job that matters.

### An analogy

Suppose you build a system to predict which hospital patients will need
emergency care, and you include in your test set the patients who are already in
the emergency room.

Your accuracy will be magnificent. You have built a very expensive way of
noticing people who are already there.

### The fix

Define the population by the cutoff:

> **Among students still enrolled on day 28, which ones will go on to fail or
> withdraw?**

Group B is excluded — not because their data is bad, but because they are not a
prediction. That is what `labels.eligible_at_cutoff()` does in
`src/oulad/labels.py`.

### What it costs

**The numbers get worse.** Noticeably.

Two reasons. The easy cases are gone. And the proportion of at-risk students
drops, because early dropouts were all at-risk by definition — removing them
removes more at-risk students than safe ones.

This is the correct outcome and worth sitting with for a second, because it is a
pattern you will meet constantly: **the fix for a methodological problem almost
always makes your results look worse.** That is what distinguishes it from a
feature. The earlier, better-looking number was measuring a task nobody wanted
done.

### What it buys

Something better than correctness: it makes the central question interesting.

The plan is to try several cutoffs — days 7, 14, 28, 56, 84 — and see how well
prediction works at each. If Group B is included, the answer is boring and
predetermined: later cutoff, more data, better score, all the way out. The chart
slopes up and says nothing.

Exclude Group B and there is a real tension, with three of the four forces
pulling against the obvious one:

| Moving the cutoff later | Effect |
|---|---|
| More days of clicks per student | Prediction gets **easier** |
| More students have already withdrawn | **Fewer people left to help** |
| The ones remaining are the ambiguous cases | Prediction gets **harder** |
| Less of the course left | **Less time to act** |

Somewhere those balance. Finding where is a genuinely useful answer — "week
five is the sweet spot, and here is what you give up by waiting until week
eight" is something a university could act on. "Accuracy increases with more
data" is not.

That trade-off is the headline result of the project. It only exists if the
population is defined correctly.

---

## 6. What this does not decide

Worth being clear about the limits of the above.

Excluding already-withdrawn students from the *prediction* does not mean
ignoring them. Their behaviour before they left is still useful training
material, and *when* people withdraw is a real finding in its own right.

There is also a more sophisticated framing that handles this whole issue
natively — modelling **time until withdrawal** rather than a yes/no outcome.
That is called survival analysis, and it is genuinely the better tool for data
shaped like this. It is out of scope for a first project, but the honest thing
is to say so in the writeup rather than pretend the binary framing was the only
option.
