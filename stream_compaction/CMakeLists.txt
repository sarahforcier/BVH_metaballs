set(SOURCE_FILES
    "common.h"
    "common.cu"
    "efficient.h"
    "efficient.cu"
    "efficient_shared.h"
    "efficient_shared.cu"
    )

cuda_add_library(stream_compaction
    ${SOURCE_FILES}
    OPTIONS -arch=sm_60
    )