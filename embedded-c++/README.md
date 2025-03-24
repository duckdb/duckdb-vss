# 🧪 Experiments

## Recall
To run experiments on uSearch, simply run
```bash
make 
```
This will run `full_coverage`, `random`, and `new_data` scenarios on all 
datasets.

The results will be output to the `usearch/results/[scenario]/` directory.

To only rebuild and run the experiments, run:
```bash
make recall
```

### Debugging
To debug the experiments, run the following command:
```bash
make all-debug
```
This will build duckdb and the experiments with debug symbols. Make sure to also
update `CMakeLists.txt` to include the debug compilaion flags and link the 
debug version of the duckdb build. When running an experiment, update the 
`Makefile` to include `cmake -DCMAKE_BUILD_TYPE=Debug` before running `make`.

This is my `launch.json` for debugging in VSCode using the `lldb` debugger:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug Tests",
            "type": "lldb",
            "request": "launch",
            "program": "${workspaceFolder}/embedded-c++/build/recall",
            "args": [],
            "cwd": "${workspaceFolder}/embedded-c++",
            "env": {
                "DUCKDB_DEBUG_PATH": "${workspaceFolder}"
            }
        }
    ]
}
```
Depending on the IDE you are using, you may need to adjust your debugger 
configuration accordingly.
