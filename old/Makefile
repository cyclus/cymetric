
ifeq ($(strip $(CYCLUS_ROOT)),)
ifneq ($(strip $(shell which cyclus)),)
CYCLUS_ROOT = $(strip $(shell cyclus --install-path))
else
CYCLUS_ROOT = $(HOME)/.local
endif
endif

includes += -I$(CYCLUS_ROOT)/include/cyclus
includes += $(shell export PKG_CONFIG_PATH="$(HOME)/.local/lib/pkgconfig"; pkg-config --cflags sqlite3)
includes += $(shell export PKG_CONFIG_PATH="$(HOME)/.local/lib/pkgconfig"; pkg-config --cflags cbc)
includes += $(shell export PKG_CONFIG_PATH="$(HOME)/.local/lib/pkgconfig"; pkg-config --cflags libxml-2.0)
includes += $(shell export PKG_CONFIG_PATH="$(HOME)/.local/lib/pkgconfig"; pkg-config --cflags libxml++-2.6)
libs += -L$(CYCLUS_ROOT)/lib
libs += -lcyclus -lhdf5 -lboost_system

cymetric: main.cc
	g++ main.cc $(includes) $(libs) -o cymetric


