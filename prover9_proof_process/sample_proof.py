_proof = """
============================== Prover9 ===============================
Prover9 (64) version 2009-11A, November 2009.
Process 72 was started by root on 1d8b1d3833ee,
Mon Mar 31 02:03:55 2025
The command was "/kaggle/working/Logic-LLM/models/symbolic_solvers/Prover9/bin/prover9".
============================== end of head ===========================

============================== INPUT =================================
assign(max_seconds,60).
clear(auto_denials).

formulas(assumptions).
(all x (Course(x) -> ContainsKnowledge(x))).
(all x all y (Takes(x,y) & Passes(x,y) -> AcquiresKnowledge(x))).
(all x (AcquiresKnowledge(x) -> GainsUnderstanding(x))).
Takes(Tuan,NLP).
Passes(Tuan,NLP).
Student(Tuan).
Course(NLP).
end_of_list.

formulas(goals).
GainsUnderstanding(Tuan).
end_of_list.

============================== end of input ==========================

============================== PROCESS NON-CLAUSAL FORMULAS ==========

% Formulas that are not ordinary clauses:
1 (all x (Course(x) -> ContainsKnowledge(x))) # label(non_clause).  [assumption].
2 (all x all y (Takes(x,y) & Passes(x,y) -> AcquiresKnowledge(x))) # label(non_clause).  [assumption].
3 (all x (AcquiresKnowledge(x) -> GainsUnderstanding(x))) # label(non_clause).  [assumption].
4 GainsUnderstanding(Tuan) # label(non_clause) # label(goal).  [goal].

============================== end of process non-clausal formulas ===

============================== PROCESS INITIAL CLAUSES ===============

% Clauses before input processing:

formulas(usable).
end_of_list.

formulas(sos).
-Course(x) | ContainsKnowledge(x).  [clausify(1)].
-Takes(x,y) | -Passes(x,y) | AcquiresKnowledge(x).  [clausify(2)].
-AcquiresKnowledge(x) | GainsUnderstanding(x).  [clausify(3)].
Takes(Tuan,NLP).  [assumption].
Passes(Tuan,NLP).  [assumption].
Student(Tuan).  [assumption].
Course(NLP).  [assumption].
-GainsUnderstanding(Tuan).  [deny(4)].
end_of_list.

formulas(demodulators).
end_of_list.

============================== PREDICATE ELIMINATION =================

Eliminating Course/1
5 Course(NLP).  [assumption].
6 -Course(x) | ContainsKnowledge(x).  [clausify(1)].
Derived: ContainsKnowledge(NLP).  [resolve(5,a,6,a)].

Eliminating Takes/2
7 Takes(Tuan,NLP).  [assumption].
8 -Takes(x,y) | -Passes(x,y) | AcquiresKnowledge(x).  [clausify(2)].
Derived: -Passes(Tuan,NLP) | AcquiresKnowledge(Tuan).  [resolve(7,a,8,a)].

Eliminating AcquiresKnowledge/1
9 -Passes(Tuan,NLP) | AcquiresKnowledge(Tuan).  [resolve(7,a,8,a)].
10 -AcquiresKnowledge(x) | GainsUnderstanding(x).  [clausify(3)].
Derived: -Passes(Tuan,NLP) | GainsUnderstanding(Tuan).  [resolve(9,b,10,a)].

Eliminating Passes/2
11 -Passes(Tuan,NLP) | GainsUnderstanding(Tuan).  [resolve(9,b,10,a)].
12 Passes(Tuan,NLP).  [assumption].
Derived: GainsUnderstanding(Tuan).  [resolve(11,a,12,a)].

Eliminating Student/1

Eliminating GainsUnderstanding/1
13 GainsUnderstanding(Tuan).  [resolve(11,a,12,a)].
14 -GainsUnderstanding(Tuan).  [deny(4)].
Derived: $F.  [resolve(13,a,14,a)].

Eliminating ContainsKnowledge/1

============================== end predicate elimination =============

Term ordering decisions:
Predicate symbol precedence:  predicate_order([ ]).
Function symbol precedence:  function_order([ ]).
After inverse_order:  (no changes).
Unfolding symbols: (none).

Auto_inference settings:
  % set(neg_binary_resolution).  % (HNE depth_diff=0)
  % clear(ordered_res).  % (HNE depth_diff=0)
  % set(ur_resolution).  % (HNE depth_diff=0)
    % set(ur_resolution) -> set(pos_ur_resolution).
    % set(ur_resolution) -> set(neg_ur_resolution).

Auto_process settings:  (no changes).

-------- Proof 1 -------- 

============================== PROOF =================================

% Proof 1 at 0.00 (+ 0.00) seconds.
% Length of proof is 12.
% Level of proof is 5.
% Maximum clause weight is 0.000.
% Given clauses 0.

2 (all x all y (Takes(x,y) & Passes(x,y) -> AcquiresKnowledge(x))) # label(non_clause).  [assumption].
3 (all x (AcquiresKnowledge(x) -> GainsUnderstanding(x))) # label(non_clause).  [assumption].
4 GainsUnderstanding(Tuan) # label(non_clause) # label(goal).  [goal].
7 Takes(Tuan,NLP).  [assumption].
8 -Takes(x,y) | -Passes(x,y) | AcquiresKnowledge(x).  [clausify(2)].
9 -Passes(Tuan,NLP) | AcquiresKnowledge(Tuan).  [resolve(7,a,8,a)].
10 -AcquiresKnowledge(x) | GainsUnderstanding(x).  [clausify(3)].
11 -Passes(Tuan,NLP) | GainsUnderstanding(Tuan).  [resolve(9,b,10,a)].
12 Passes(Tuan,NLP).  [assumption].
13 GainsUnderstanding(Tuan).  [resolve(11,a,12,a)].
14 -GainsUnderstanding(Tuan).  [deny(4)].
15 $F.  [resolve(13,a,14,a)].

============================== end of proof ==========================

============================== STATISTICS ============================

Given=0. Generated=1. Kept=0. proofs=1.
Usable=0. Sos=0. Demods=0. Limbo=0, Disabled=13. Hints=0.
Kept_by_rule=0, Deleted_by_rule=0.
Forward_subsumed=0. Back_subsumed=0.
Sos_limit_deleted=0. Sos_displaced=0. Sos_removed=0.
New_demodulators=0 (0 lex), Back_demodulated=0. Back_unit_deleted=0.
Demod_attempts=0. Demod_rewrites=0.
Res_instance_prunes=0. Para_instance_prunes=0. Basic_paramod_prunes=0.
Nonunit_fsub_feature_tests=0. Nonunit_bsub_feature_tests=0.
Megabytes=0.03.
User_CPU=0.00, System_CPU=0.00, Wall_clock=0.

============================== end of statistics =====================

============================== end of search =========================

THEOREM PROVED

THEOREM PROVED

Exiting with 1 proof.

------ process 72 exit (max_proofs) ------

Process 72 exit (max_proofs) Mon Mar 31 02:03:55 2025
"""
