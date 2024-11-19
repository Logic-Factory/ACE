# Technology Library

### Standard Cell Library

- Merge the original .lib files into a single .lib file

    ```
    This tool is provided in iFlow or OpenROAD.
    ```

- convert the .lib file to .genlib by ABC

    ```
    read <merged_liberty_file>
    write_genlib <genlib_file>
    ```

### Gtech library

    *gtech.genlib* is the defined generic technology library for the LogicFactory project to transform the input RTL into the intermediate representation by defined gtech.
