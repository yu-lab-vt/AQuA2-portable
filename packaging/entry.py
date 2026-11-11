"""Windows GUI bootloader. Freeze support MUST precede heavy imports."""
if __name__=='__main__':
    from aqua2.portable import configure_runtime
    configure_runtime()
    import multiprocessing
    multiprocessing.freeze_support()
    from aqua2.portable import main
    raise SystemExit(main())
