.PHONY: build install debug clean

EXTENSION_DIR := $(shell pwd)
VSIX := completely-0.0.1.vsix
PROFILE := Blank

build:
	npm run package
	npx @vscode/vsce package --no-dependencies --allow-missing-repository

install: build
	code --profile $(PROFILE) --install-extension $(VSIX) --force

debug: install
	code --profile $(PROFILE) --new-window ..

clean:
	rm -rf dist out *.vsix
