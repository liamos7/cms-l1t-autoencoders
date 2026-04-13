import h5py
import numpy as np

input_file  = "launchable/ZB_17Apr2025_0000.h5"
output_file = "launchable/ZB_npvgood_geq10.h5"
CHUNK_SIZE  = 100_000

with h5py.File(input_file, "r", locking=False) as fin, \
     h5py.File(output_file, "w") as fout:

    n_events = fin["PV_npvsGood"].shape[0]

    # ── Cache all references once (avoids repeated dict lookups) ──────────
    in_dsets = {name: fin[name] for name in fin}
    out_dsets = {
        name: fout.create_dataset(
            name,
            shape=(0,) + ds.shape[1:],
            maxshape=(None,) + ds.shape[1:],
            dtype=ds.dtype,
            chunks=True,
            compression="lzf",       # fast byte-shuffle compression
            shuffle=True,
        )
        for name, ds in in_dsets.items()
    }

    pv_in       = in_dsets["PV_npvsGood"]
    write_index = 0

    for start in range(0, n_events, CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, n_events)

        # ── Read PV first; skip everything else if no events pass ─────────
        mask = pv_in[start:stop] >= 10
        keep = int(np.count_nonzero(mask))   # faster than np.sum on bool
        if keep == 0:
            continue

        new_size = write_index + keep
        for name, in_ds in in_dsets.items():
            out_ds = out_dsets[name]
            out_ds.resize(new_size, axis=0)                  # one resize per dataset
            out_ds[write_index:new_size] = in_ds[start:stop][mask]

        write_index += keep

print(f"Done — kept {write_index:,} / {n_events:,} events "
      f"({100*write_index/n_events:.1f}%)")