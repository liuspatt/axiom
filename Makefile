PRIV_DIR := $(MIX_APP_PATH)/priv
NIF_PATH := $(PRIV_DIR)/libpython_interface.so
C_SRC := $(shell pwd)/c_src

CPPFLAGS := -shared -fPIC -fvisibility=hidden -std=c++17 -Wall -Wextra -Wno-unused-parameter -Wno-comment
CPPFLAGS += -I$(ERTS_INCLUDE_DIR) -I$(FINE_INCLUDE_DIR)

ifdef DEBUG
	CPPFLAGS += -g
else
	CPPFLAGS += -O3
endif

ifndef TARGET_ABI
  TARGET_ABI := $(shell uname -s | tr '[:upper:]' '[:lower:]')
endif

ifeq ($(TARGET_ABI),darwin)
	CPPFLAGS += -undefined dynamic_lookup -flat_namespace
endif

SOURCES := $(wildcard $(C_SRC)/*.cpp)
HEADERS := $(wildcard $(C_SRC)/*.hpp)

all: $(NIF_PATH)
	@ echo > /dev/null # Dummy command to avoid the default output "Nothing to be done"

$(NIF_PATH): $(SOURCES) $(HEADERS)
	@ mkdir -p $(PRIV_DIR)
	$(CXX) $(CPPFLAGS) $(SOURCES) -o $(NIF_PATH)
