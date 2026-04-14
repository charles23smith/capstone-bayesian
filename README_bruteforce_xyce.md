# Brute Force + Xyce Workflow

## General Steps

1. Run `bruteForce.py` for the shot you want:

```powershell
python bruteForce.py --csv data/test#_data.csv --out foldername
```

Example:

```powershell
python bruteForce.py --csv data/27271_data.csv --out diode_fit_outputs_27271
```

2. After `bruteForce.py` finishes, take the exported current source file:

`export_current_pulse.txt`

and place it in a folder together with:

- the desired `.cir` file
- the desired test data / Xyce run files for that shot

3. Open a Xyce terminal, `cd` into that folder, and run Xyce on the circuit file:

```powershell
"C:\Program Files\XyceNF_7.10\bin\Xyce.exe" shot####.cir
```

Example:

```powershell
"C:\Program Files\XyceNF_7.10\bin\Xyce.exe" shot27271.cir
```

4. Update the two path lines at the top of `xycecompare.py` so they point to the correct files for the shot you want to compare.

You need to set:

- `SIM_FILE`
- `EXP_FILE`

5. Run the comparison script:

```powershell
python xycecompare.py
```

## General Template

For any test:

```powershell
python bruteForce.py --csv data/TESTNUMBER_data.csv --out OUTPUT_FOLDER
```

Then copy `export_current_pulse.txt` into the Xyce folder with the matching `.cir` file, and run:

```powershell
"C:\Program Files\XyceNF_7.10\bin\Xyce.exe" shotTESTNUMBER.cir
python xycecompare.py
```

## Full Example: Shot 27271

Run brute force:

```powershell
python bruteForce.py --csv data/27271_data.csv --out diode_fit_outputs_27271
```

Move `diode_fit_outputs_27271/export_current_pulse.txt` into the folder that contains:

- `shot27271.cir`
- the Xyce files for shot `27271`

In a Xyce terminal:

```powershell
cd E:\CapstoneXyce
"C:\Program Files\XyceNF_7.10\bin\Xyce.exe" shot27271.cir
```

Then set the top of `xycecompare.py` to the matching `27271` simulation and experimental files, and run:

```powershell
python xycecompare.py
```
