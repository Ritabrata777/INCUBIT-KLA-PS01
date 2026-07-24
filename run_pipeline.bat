@echo off
REM run_pipeline.bat
REM
REM Chains training (self-training / on-the-fly synthetic degradation) and
REM batched evaluation into a single command for NAFNet-SR semiconductor
REM image restoration.
REM
REM Usage:
REM   run_pipeline.bat
REM
REM Override any setting by setting an env var before running, e.g.:
REM   set BATCH_SIZE=32
REM   set UPSCALE_FACTOR=2
REM   run_pipeline.bat
REM
REM To skip training and only run evaluation with existing weights:
REM   set SKIP_TRAIN=1
REM   run_pipeline.bat

setlocal enabledelayedexpansion

if not defined CLEAN_DIR set CLEAN_DIR=data\clean
if not defined VAL_CLEAN_DIR set VAL_CLEAN_DIR=data\val_clean
if not defined TEST_INPUT_DIR set TEST_INPUT_DIR=data\test_input
if not defined OUTPUT_DIR set OUTPUT_DIR=data\restored_output
if not defined WEIGHTS_PATH set WEIGHTS_PATH=final_model_weights.pt

if not defined EPOCHS set EPOCHS=100
if not defined BATCH_SIZE set BATCH_SIZE=16
if not defined PATCH_SIZE set PATCH_SIZE=128
if not defined UPSCALE_FACTOR set UPSCALE_FACTOR=4
if not defined WIDTH set WIDTH=32
if not defined NUM_BLOCKS set NUM_BLOCKS=8
if not defined NUM_WORKERS set NUM_WORKERS=8

if not defined EVAL_BATCH_SIZE set EVAL_BATCH_SIZE=32
if not defined SKIP_TRAIN set SKIP_TRAIN=0

echo ==============================================================
echo  NAFNet-SR pipeline
echo ==============================================================
echo  clean_dir          : %CLEAN_DIR%
echo  val_clean_dir      : %VAL_CLEAN_DIR%
echo  test_input_dir     : %TEST_INPUT_DIR%
echo  output_dir         : %OUTPUT_DIR%
echo  weights_path       : %WEIGHTS_PATH%
echo  epochs             : %EPOCHS%
echo  batch_size (train) : %BATCH_SIZE%
echo  batch_size (eval)  : %EVAL_BATCH_SIZE%
echo  patch_size         : %PATCH_SIZE%
echo  upscale_factor     : %UPSCALE_FACTOR%
echo  width / num_blocks : %WIDTH% / %NUM_BLOCKS%
echo ==============================================================

if "%SKIP_TRAIN%"=="1" (
    echo.
    echo ----- Step 1/2: Skipped ^(SKIP_TRAIN=1^) -----
) else (
    echo.
    echo ----- Step 1/2: Training ^(self-training mode^) -----
    python train.py ^
        --self_train ^
        --clean_dir "%CLEAN_DIR%" ^
        --val_clean_dir "%VAL_CLEAN_DIR%" ^
        --epochs %EPOCHS% ^
        --batch_size %BATCH_SIZE% ^
        --patch_size %PATCH_SIZE% ^
        --upscale_factor %UPSCALE_FACTOR% ^
        --width %WIDTH% ^
        --num_blocks %NUM_BLOCKS% ^
        --num_workers %NUM_WORKERS% ^
        --output_path "%WEIGHTS_PATH%"
    if errorlevel 1 (
        echo Training failed. Aborting.
        exit /b 1
    )
)

echo.
echo ----- Step 2/2: Batched FP16 evaluation -----
python evaluation.py ^
    --input_dir "%TEST_INPUT_DIR%" ^
    --output_dir "%OUTPUT_DIR%" ^
    --weights_path "%WEIGHTS_PATH%" ^
    --batch_size %EVAL_BATCH_SIZE% ^
    --upscale_factor %UPSCALE_FACTOR% ^
    --width %WIDTH% ^
    --num_blocks %NUM_BLOCKS% ^
    --num_workers %NUM_WORKERS%
if errorlevel 1 (
    echo Evaluation failed.
    exit /b 1
)

echo.
echo Done. Restored images saved to %OUTPUT_DIR%
endlocal
