# VSAG (Vector Search for AntGroup)

## Dependencies
### Ubuntu
```
$ ./scripts/deps/install_deps_ubuntu.sh
```

### AliOS/Centos
```
$ ./scripts/deps/install_deps_centos.sh
```

## Usage
```bash
Usage: make <target>

Targets:
help:                   ## Show the help.
debug:                  ## Build vsag with debug options.
release:                ## Build vsag with release options.
format:                 ## Format codes.
test:                   ## Build and run unit tests.
test_asan:              ## Build and run unit tests with AddressSanitizer option.
test_cov:               ## Build and run unit tests with code coverage enabled.
benchmark:              ## Run benchmarks.
clean:                  ## Clear build/ directory.
install: debug          ## Build and install the debug version of vsag.
```

## Roadmap
