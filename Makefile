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
TEST_C_SRC   = $(wildcard $(TEST_DIR)/*.c)
ALL_TEST_SRC = $(wildcard $(TEST_DIR)/*.c $(TEST_DIR)/*.cu)

# === Objects ===
MODULE_OBJ   = $(patsubst $(MODULE_DIR)/%.cu, $(OBJ_DIR)/%.o, $(MODULE_SRC))
OPT_OBJ      = $(patsubst $(OPT_DIR)/%.cu, $(OBJ_DIR)/%.o, $(OPT_SRC))
TOOL_CU_OBJ  = $(patsubst $(TOOL_DIR)/%.cu, $(OBJ_DIR)/%.o, $(TOOL_CU_SRC))
TEST_C_OBJ   = $(patsubst $(TEST_DIR)/%.c, $(OBJ_DIR)/%.o, $(TEST_C_SRC))

TEST_NAME ?= test_main
EXEC      = $(TEST_NAME)

.PHONY: all test clean format

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

format:
	clang-format -i $(MODULE_SRC) $(OPT_SRC) $(TOOL_CU_SRC) $(ALL_TEST_SRC)

clean:
	rm -rf $(OBJ_DIR) $(EXEC) $(TEST_DIR)/*.o
