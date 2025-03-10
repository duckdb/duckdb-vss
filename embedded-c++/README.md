# 🧪 Experiments

## Recall
To run experiments on recall and unreachable points, simply run 
```bash
make all
```
This will run the experiments defined in the `recall.cpp` file.

The results will be output to a CSV file with name 
`[dataset]_merged_report.csv` in this directory.

To only rebuild and run the experiments, run:
```bash
make recall
```

For now, only the `fashion-mnist` and `mnist` are ran. To run other datasets,
include a runner for the given dataset in the `recall.cpp` file, e.g. after 
line 326:
```cpp
//Run test on sift
runner.runTest(2);
//Run test on gist
runner.runTest(3);
```
### Limitations
- The experiments are run on a single thread, more checks should be made to 
    ensure that running experiments on multiple threads don't cause any 
    unwanted behaviour impacting experiment results.
- Only 119 iterations of delete/insert 600 vectors can be done due to an
    unknown issue with the `free_keys_` ring buffer (delete list). After 
    enough operations utilizing the delete list, the program will hang. If more 
    vectors are deleted/inserted, less iterations are possible. If more 
    iterations are needed, lower the sample size in the `recall.cpp` file (line 
    113).
- The current implementation uses a modified ring buffer implementation. More 
    tests should be done to ensure that the ring buffer implementation is 
    correct and doesn't cause any unwanted behaviour.

### Debugging
To debug the experiments, run the following command:
```bash
make all-debug
```
This will build duckdb and the experiments with debug symbols. Make sure to also
update `CMakeLists.txt` to include the debug compilaion flags and link the 
debug version of the duckdb build.

To only build the experiments with debug symbols, run:
```bash
make recall-debug
```

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
