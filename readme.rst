Description
===========

DeCross is a spatio-temporal cross color (rainbow) reduction filter.

DeCross must be used right after the source filter, before any field
matching or deinterlacing.

The luma is returned unchanged.

Currently only x86 systems are supported.

This is a port of the DeCross Avisynth plugin, which is in turn based
on the AviUtl plugin CrossColor.


Usage
=====
::

    decross.DeCross(clip clip, [int thresholdy=30, int noise=60, int margin=1, bint debug=False])


Parameters:
    *clip*
        A clip to process. It must have constant format and dimensions
        and it must be YUV420P8 or YUV422P8.

    *thresholdy*
        Edge detection threshold. Must be between 0 and 255.

        Smaller values will detect and filter more edges.

        Default: 30.

    *noise*
        Luma difference threshold. Must be between 0 and 255.

        Smaller values will filter more conservatively.

        Default: 60.

    *margin*
        Expands the edge mask to the left and right by *margin*
        pixels. Must be between 0 and 4.

        Default: 1.

    *debug*
        Instead of filtering the rainbows, show the areas that would
        be filtered.

        Default: False.


Compilation
===========

::

    mkdir build && cd build
    meson ../
    ninja


License
=======

GNU GPL v2, like the Avisynth plugin.
