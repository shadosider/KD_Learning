# ============================================================
# KD 实验一键流水线
#
# 用法:
#   make                 # 浅模式(shallow teacher)顺序执行全部 5 个操作
#   make DEEP=1          # 深模式(deep teacher + dropout),产物写 _deep 后缀
#   make DO_ABLATION=0   # 跳过消融实验,其余照常
#   make kd DEEP=1       # 只重跑蒸馏(需 teacher_deep.pt 与 student_init.pt 已存在)
#
# 开关(默认全 1 = 执行):
#   DO_TEACHER  DO_STUDENT  DO_KD  DO_VIZ  DO_ABLATION
#
# 依赖顺序:teacher -> student(baseline) -> kd -> viz / ablation
# student_init.pt 由 student 阶段内部先生成;student 与 deep 无关,复用同一份产物
# ============================================================

DEEP ?= 0
DO_TEACHER ?= 1
DO_STUDENT ?= 1
DO_KD ?= 1
DO_VIZ ?= 1
DO_ABLATION ?= 1

PY := ./.venv/bin/python

ifeq ($(DEEP),1)
DEEP_FLAG := --deep
else
DEEP_FLAG :=
endif

.PHONY: all teacher student kd viz ablation

# all 必须是第一个 target,作为 make 的默认目标
all:
	@if [ "$(DO_TEACHER)" = "1" ]; then $(MAKE) teacher; fi
	@if [ "$(DO_STUDENT)" = "1" ]; then $(MAKE) student; fi
	@if [ "$(DO_KD)" = "1" ]; then $(MAKE) kd; fi
	@if [ "$(DO_VIZ)" = "1" ]; then $(MAKE) viz; fi
	@if [ "$(DO_ABLATION)" = "1" ]; then $(MAKE) ablation; fi

teacher:
	$(PY) experiments/train_teacher.py $(DEEP_FLAG)

student:
	$(PY) experiments/student_init.py
	$(PY) experiments/train_student.py

kd:
	$(PY) experiments/distill_student.py $(DEEP_FLAG)

viz:
	$(PY) experiments/visualize_results.py $(DEEP_FLAG)

ablation:
	$(PY) experiments/ablation_study.py $(DEEP_FLAG)
