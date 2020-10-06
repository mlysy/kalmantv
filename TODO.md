# TODO

- [ ] Naming conventions: Plural of argument names, maybe something more memorable than `gss`?

	For `gss`, let's go with `genss`.
	
- [ ] Unit tests should be against `genss`.  There's a way to use `genss` to create inputs to `filter`, `smooth`, etc. that doesn't involve running the Kalman filter/smoother for testing.

- [ ] Input checks for `*.pyx` files.  Ideally should accept non-Fortran arrays as well, with copy inside methods only if needed.

- [ ] More information on installation, e.g., Eigen path, flags.
