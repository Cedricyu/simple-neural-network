# === Config ===
NVCC      = nvcc
CC        = gcc
CFLAGS    = -I./module -I./optimizer -I./tool -I./test
NVCCFLAGS = -I./module -I./optimizer -I./tool -I./test
LDFLAGS   = -lm

# === Directories ===
MODULE_DIR = module
OPT_DIR    = optimizer
TOOL_DIR   = tool
TEST_DIR   = test
OBJ_DIR    = build

# === Sources ===
MODULE_SRC   = $(wildcard $(MODULE_DIR)/*.cu)
OPT_SRC      = $(wildcard $(OPT_DIR)/*.cu)
TOOL_CU_SRC  = $(wildcard $(TOOL_DIR)/*.cu)
# train_worker.c and train_viewer.c each have their own main() (see the
# comment atop train_worker.c for why they're two separate executables), so
# both are excluded from the test_main build below and compiled on their own.
LIVE_GUI_SRC = $(TEST_DIR)/train_worker.c $(TEST_DIR)/train_viewer.c
TEST_C_SRC   = $(filter-out $(LIVE_GUI_SRC), $(wildcard $(TEST_DIR)/*.c))
ALL_TEST_SRC = $(wildcard $(TEST_DIR)/*.c $(TEST_DIR)/*.cu)

# === Objects ===
MODULE_OBJ   = $(patsubst $(MODULE_DIR)/%.cu, $(OBJ_DIR)/%.o, $(MODULE_SRC))
OPT_OBJ      = $(patsubst $(OPT_DIR)/%.cu, $(OBJ_DIR)/%.o, $(OPT_SRC))
TOOL_CU_OBJ  = $(patsubst $(TOOL_DIR)/%.cu, $(OBJ_DIR)/%.o, $(TOOL_CU_SRC))
TEST_C_OBJ   = $(patsubst $(TEST_DIR)/%.c, $(OBJ_DIR)/%.o, $(TEST_C_SRC))

TEST_NAME ?= test_main
EXEC        = $(TEST_NAME)
WORKER_EXEC = train_worker
VIEWER_EXEC = train_viewer

# Only used by the live-training targets; harmless if pkg-config/SDL2 aren't
# installed since nothing else depends on these.
SDL2_CFLAGS = $(shell pkg-config --cflags sdl2 SDL2_ttf 2>/dev/null)
SDL2_LIBS   = $(shell pkg-config --libs sdl2 SDL2_ttf 2>/dev/null)

.PHONY: all test clean format train_gui

all: $(EXEC)
	@echo "===== Build Complete: $(EXEC) ====="

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(OBJ_DIR)/%.o: $(MODULE_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(OPT_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(TOOL_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(TEST_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(EXEC): $(MODULE_OBJ) $(OPT_OBJ) $(TOOL_CU_OBJ) $(TEST_C_OBJ)
	$(NVCC) -o $@ $^ $(LDFLAGS)

test: $(EXEC)
	./$(EXEC)

# train_worker is plain CUDA/C (no SDL2) - built and linked with $(NVCC) like
# everything else.
$(OBJ_DIR)/train_worker.o: $(TEST_DIR)/train_worker.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(WORKER_EXEC): $(MODULE_OBJ) $(OPT_OBJ) $(TOOL_CU_OBJ) $(OBJ_DIR)/mnist.o $(OBJ_DIR)/train_worker.o
	$(NVCC) -o $@ $^ $(LDFLAGS)

# train_viewer is plain SDL2 (no CUDA headers or libraries at all) - built
# and linked with $(CC) only, deliberately never touching $(NVCC).
$(OBJ_DIR)/train_viewer.o: $(TEST_DIR)/train_viewer.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) $(SDL2_CFLAGS) -c $< -o $@

$(VIEWER_EXEC): $(OBJ_DIR)/train_viewer.o
	$(CC) -o $@ $^ $(SDL2_LIBS) -lm

# Runs both halves of the live-training UI together: the SDL2 viewer in the
# background, the CUDA trainer in the foreground. They hand off over a FIFO
# (see train_record.h) rather than sharing a process - see the comment atop
# train_worker.c for why.
train_gui: $(WORKER_EXEC) $(VIEWER_EXEC)
	./$(VIEWER_EXEC) & \
	viewer_pid=$$!; \
	./$(WORKER_EXEC); \
	wait $$viewer_pid

format:
	clang-format -i $(MODULE_SRC) $(OPT_SRC) $(TOOL_CU_SRC) $(ALL_TEST_SRC)

clean:
	rm -rf $(OBJ_DIR) $(EXEC) $(WORKER_EXEC) $(VIEWER_EXEC) $(TEST_DIR)/*.o
