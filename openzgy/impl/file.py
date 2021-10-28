#!/usr/bin/env python3

"""
Low level I/O, abstract layer.

This file contains classes that do low level I/O either to on-prem data
using the regular read and write methods of the OS or to seismic store
using SDAPI. Other back ends can easily be added.

Some work is done to consolidate read requests and to buffer write
requests in the cloud back end. This is needed to get acceptable
performance. This explains why this file is quite large.


    impl.file.Config:
      impl.file.FileConfig(Config):
      impl.file.SDConfig(Config):

        * Details such as user credentials etc. established when the
          file is open. Specific to the backend type.
        * Note that there is currently no way to pass a configuration
          object along with every read and write request. This might
          have been useful for a server type application but would
          require the config parameter to ripple across at least 50
          existing methods. I doubt this would be worth the trouble.

    impl.file.FileADT:
      impl.file.LocalFile(FileADT):
        impl.file.LocalFileOther(LocalFile):
        impl.file.LocalFileLinux(LocalFile):
      impl.file.SeismicStoreFile(FileADT):
      impl.file.SeismicStoreFileDelayedWrite(FileADT):

        * Higher level code should only access the polymorphic FileADT
          base class and the impl.file.FileFactory that creates an
          instance of the desired type.
"""
##@package openzgy.impl.file
#@brief Low level I/O, abstract layer.

import os
import json
from enum import Enum
from contextlib import contextmanager, suppress
from collections import namedtuple

from ..exception import *
import warnings

try:
    import sdglue as sd
except Exception as ex:
    warnings.warn("seismic store access is not available: " + str(ex))
    sd = None

class UsageHint(Enum):
    Unknown    = 0x00
    TextFile   = 0x01
    Header     = 0x10
    Data       = 0x20
    Compressed = 0x40,

class Config:

    _deprecation_context_warned = False
    @staticmethod
    def _deprecated_context(attrname):
        if not Config._deprecation_context_warned:
            print('DEPRECATION WARNING: IOContext should be a dict. Found "' +
                  attrname + '" as an attribute.')
            Config._deprecation_context_warned = True

    @staticmethod
    def  _get_string_env(context, attrname, name, default):
        try:
            value = context[attrname]
            name = attrname
        except (TypeError, KeyError):
            if hasattr(context, attrname):
                value = getattr(context, attrname)
                name = attrname
                Config._deprecated_context(attrname)
            else:
                value = os.getenv(name, str(default))
        if default and not value:
            print("WARNING: ${0} cannot be empty.".format(name))
            value = default
        return value

    @staticmethod
    def  _get_numeric_env(context, attrname, name, default, min_value, max_value):
        try:
            value = context[attrname]
            name = attrname
        except (TypeError, KeyError):
            if hasattr(context, attrname):
                value = getattr(context, attrname)
                name = attrname
                Config._deprecated_context(attrname)
            else:
                try:
                    value = int(os.getenv(name, str(default)))
                except ValueError:
                    print("WARNING: badly formed number in ${0} ignored.".format(name))
                    value = default
        if value < min_value or value > max_value:
            print("WARNING: ${0} must be be between {1} and {2}.".format(name, min_value, max_value))
            value = default
        return value

class FileConfig(Config):

    def __init__(self, context):
        """
        Currently the on-prem file reader has no user settable configurfation.
        """
        self.maxsize = 0 # No consolidation of read requests.
        self.maxhole = 0 # No fuzzy consolidation.
        self.aligned = 0 # No additional alignment on reads.
        self.segsize = 0 # No buffering of writes.
        self.threads = 1 # No multi threading.

    @staticmethod
    def _redact(s):
        return s if len(s) < 20 else s[:4] + "..." + s[-4:]

    def dump(self):
        return """FileConfig: No user settable configuration."""

class SDConfig(Config):

    def __init__(self, context):
        """
        Process an iocontext for seismic store, doing consistency checks
        and applying fallbacks from environment variables and hard coded
        defaults.

        The context itself should be a dict or equivalent. The code also
        supports the older style using a class instance with attributes.
        That feature is deprecated and will be removed. It violates the
        principle of least surprise.

        A fully specified iocontext has the following attributes:

            sdurl: string
                Where to contact the seismic store service.
                Defaults to $OPENZGY_SDURL.

            sdapikey: string
                Authorization for application to access the seismic store API.
                Defaults to $OPENZGY_SDAPIKEY.

            sdtoken: string
                User credentials. Set to $OPENZGY_TOKEN if not found,
                beware that this might not be secure. The code will no longer
                use the token last saved by sdcfg as a fallback. If this is
                desired you must specify "FILE:carbon.slbapp.com" as the token.
                Caveat: The sdcfg token is not refreshed so it might time out
                after an hour. Run "sdutil auth idtoken > /dev/null" to refresh.

            maxsize: int specified in MB between 0 and 1024.
                Zero is taken to mean do not consolidate.
                Tell the reader to try to consolidate neighboring bricks
                when reading from seismic store. This is usually possible
                when the application requests full traces or at least traces
                traces longer then 64 samples. Setting maxsize limits this
                consolidation to the specified size. The assumption is that
                for really large blocks the per-block overhead becomes
                insignificant compared to the transfer time.

                Consolidating requests has higher priority than using
                multiple threads. So, capping maxsize might allow more
                data to be read in parallel.

                Note that currently the spitting isn't really smart. With a
                64 MB limit and 65 contiguous 1 MB buffers it might end up
                reading 64+1 MB instead of e.g. 32+33 MB.

                Note that the low level reader should not assume that
                requests are capped at this size. They might be larger
                e.g. when reading the header information.

                Defaults to $OPENZGY_MAXSIZE_MB if not specified, or 2 MB.

            maxhole: int specified in MB between 0 and 1024.
                This applies when consolidate neighboring bricks when
                reading from seismic store. Setting maxhole > 0 tells the
                reader that it is ok to also consolidate requests that are
                almost neighbors, with a gap up to and including maxhole.
                The data read from the gap will be discarded unless picked
                up by some (not yet implemented) cache.

                For cloud access with high bandwidth (cloud-to-cloud) this
                should be at least 2 MB because smaller blocks will take
                just as long to read. For low bandwidth cloud access
                (cloud-to-on-prem) it should be less. If a fancy cache
                is implemented it should be more. For accessing on-prem
                ZGY files it probably makes no difference.
                Defaults to $OPENZGY_MAXHOLE_MB if not specified, or 2 MB.

            aligned: int in MB between 0 and 1024.
                This is similar to the maxhole parameter. If set, starting
                and ending offsets are extended so they both align to the
                specified value. Set this parameter if the lower levels
                implement a cache with a fixed blocksize and when there is
                an assumpton that most reads will be aligned anyway.
                TODO-Worry: Handling reads past EOF may become a challenge
                for the implementation.
                Defaults to $OPENZGY_ALIGNED_MB if not specified, or zero.

            segsize: int in MB between 0 and 16*1024 (i.e. 16 GB).
                Defaults to $OPENZGY_SEGSIZE_MB if not specified, or 1 GB.

            threads: int between 1 and 1024.
                Use up to this many parallel requests to seismic store
                in order to speed up processing. This applies to individual
                reads in the main API. So the reads must be for a large
                area (i.e. covering many bricks) for the setting to be
                of any use. Set to $OPENZGY_NUMTHREADS if not found,
                and 1 (i.e. no threading) if the environment setting is
                also missing.

                Whether it is useful to set the variable depende on the
                application. Apps such as Petrel/BASE generally do their
                own multi threading, issuing multiple read requests to
                the high level API in parallel. In that case it might
                not be useful to also parallelize individual requests.

            legaltag: string, possibly empty.
                The legaltag stored in the file. Used only on create.

            writeid:
                I don't know what this is for. Ask the seismic store team.

            seismicmeta:
                a dictionary of additional information to be associated
                with this dataset in the data ecosystem. Currently used
                only on create, although SDAPI allows this to be set on
                an existing file by calling {get,set}SeismicMeta().

                When set via an environment variable (strictly for testing)
                this needs to be the string representation of the json data.
                When set from a program a Python dict is expected.

            _debug_trace:
                For debugging and unit tests only.
                Callback to be invoked immediately before a read or write
                is passed on to seismic store. Typically used to verify
                that consolidating bricks works as expected. Can only be
                set programmatically. Not by an environment variable.
"""
        self.sdurl    = self._get_string_env(context, "sdurl",
                            "OPENZGY_SDURL",
                            "")
        self.sdapikey = self._get_string_env(context, "sdapikey",
                            "OPENZGY_SDAPIKEY",
                            "")
        self.sdtoken  = self._get_string_env(context, "sdtoken",
                            "OPENZGY_TOKEN",
                            "")
        self.maxsize = self._get_numeric_env(context, "maxsize",
                            "OPENZGY_MAXSIZE_MB",64,0,1024) * (1024*1024)
        self.maxhole = self._get_numeric_env(context, "maxhole",
                            "OPENZGY_MAXHOLE_MB",2,0,1024) * (1024*1024)
        self.aligned = self._get_numeric_env(context, "aligned",
                            "OPENZGY_ALIGNED_MB",0,0,1024) * (1024*1024)
        self.segsize = self._get_numeric_env(context, "segsize",
                            "OPENZGY_SEGSIZE_MB",1024,0,1024*16) * (1024*1024)
        self.threads = self._get_numeric_env(context, "threads",
                            "OPENZGY_NUMTHREADS",1,1,1024)
        # All the numeric options ought to be integral, but for unit
        # tests it might be desirable to allow odd sizes. When reading
        # from environment variables only integral numbers are accepted.
        self.maxsize = int(round(self.maxsize))
        self.maxhole = int(round(self.maxhole))
        self.aligned = int(round(self.aligned))
        self.segsize = int(round(self.segsize))

        self.legaltag = self._get_string_env(
            context, "legaltag", "OPENZGY_LEGALTAG", "")
        self.writeid = self._get_string_env(
            context, "writeid", "OPENZGY_WRITEID", "")
        self.seismicmeta = self._get_string_env(
            context, "seismicmeta", "OPENZGY_SEISMICMETA", "")

        try:
            self._debug_trace = context["_debug_trace"]
        except (TypeError, KeyError):
            if hasattr(context, "_debug_trace"):
                self._debug_trace = getattr(context, "_debug_trace")
            else:
                self._debug_trace = lambda *args, **kwargs: False

    @property
    def extra(self):
        """
        Legaltag, writeid, and seismicmeta are usually packed
        into a single "extra" dictionary when creating a new file.
        If any of them are unset they will be excluded from the
        dictionary instead of being passed as some default value.

        CAVEAT: The keys in the "extra" dictionary are not
        supposed to be hard coded as I do here. They are defined in
        seismic-store-client-api-cpp/src/src/core/Constants.{cc,h}.
        Cannot access that file here.

        NOTE: Python dicts have an undefined sort order, as does
        json. To simplify testing I sort the keys in the "extra" dict.
        If SDAPI for some reason should require a specific ordering
        then "seismicmeta" needs to be passed as a string.
        """
        result = {}
        if self.legaltag:
            result["legal-tag"] = self.legaltag
        if self.writeid:
            result["writeid"] = self.writeid
        if self.seismicmeta:
            if not isinstance(self.seismicmeta, str):
                result["seismicmeta"] = json.dumps(
                    self.seismicmeta, sort_keys=True)
            else:
                result["seismicmeta"] = self.seismicmeta
        return result

    @staticmethod
    def _redact(s):
        return s if len(s) < 20 else s[:4] + "..." + s[-4:]

    def dump(self):
        return """SDConfig:
   sdurl:    {sdurl}
   sdapikey: {sdapikey}
   sdtoken:  {sdtoken}
   maxsize:  {maxsize} MB
   maxhole:  {maxhole} MB
   aligned:  {aligned} MB
   segsize:  {segsize} MB
   threads:  {threads}
   extra:    {extra}""".format(
       sdurl=self.sdurl,
       sdapikey=self._redact(self.sdapikey),
       sdtoken=self._redact(self.sdtoken),
       maxsize=self.maxsize // (1024*1024),
       maxhole=self.maxhole // (1024*1024),
       aligned=self.aligned // (1024*1024),
       segsize=self.segsize // (1024*1024),
       threads=self.threads,
       extra=json.dumps(self.extra, sort_keys=True))

class FileADT:

    def __init__(self, filename, mode, iocontext):
        """
        Open a file in the specified mode, which must be "rb" or "w+b".
        Caller should use a "with" block to ensure the file gets closed.
        The iocontext is an optional data structure that the user may
        specify when a reader is created. It might be used to hold
        user credentials etc. needed to access the low level file.
        TODO-Low: support "r+b" (update) at some point in the future.
        """
        if not mode in ("rb", "w+b", "r+b", "d"):
            raise ZgyUserError("Opening ZGY as " + mode + " is not supported.")
        # Currently no need to keep this, as the derived classes will
        # copy the relevant information to self._config.
        #self._iocontext = iocontext

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.xx_close()

    def xx_close(self):
        """
        Close a previously opened file.
        No action if the file is already closed.
        """
        raise NotImplementedError

    def xx_read(self, offset, size, *, usagehint=UsageHint.Unknown):
        """
        Read binary data from the file. Both size and offset are mandatory.
        I.e. caller is not allowed to read "the entire file", and not
        allowed to "read from where I left off the last time".
        The actual reading will be done in a derived class.
        The base class only validates the arguments.
        """
        raise NotImplementedError

    def xx_write(self, data, offset, *, usagehint=UsageHint.Unknown):
        """
        Write binary data to the file. Offset is mandatory. I.e. caller
        is not allowed to "write to where I left off the last time".
        The actual writing will be done in a derived class.
        The base class only validates the arguments.
        """
        raise NotImplementedError

    # Might want to pass this tuple to the delivery functor instead of
    # discrete arguments. Probably doesn't change much either way, though.
    # Or maybe consider passing everything as keyword arguments.
    _deliveryType = namedtuple("Delivery", "data this_offset this_size want_offset want_size iocontext")
    _requestType = namedtuple("Request", "offset size delivery")

    def xx_readv(self, requests, *, parallel_ok=False, immutable_ok=False, transient_ok=False, usagehint=UsageHint.Unknown):
        """
        Read binary data from multiple regions in the file. Each part
        of the request specifies offset, size, and a delivery functor
        which will be invoked to pass back the returned bulk.

        Arguments:
            parallel_ok:  If true then the delivery functor might be called
                          simultaneously from multiple worker threads.
                          The function itself will block until all the data
                          has been read or an error occurs.
            immutable_ok: If true the caller promises that the delivery
                          functor will not try to modify the data buffer.
                          Pass False e.g. if the functor may need to byteswap
                          the data it has read from file.
            transient_ok: If true the caller promises that the delivery
                          functor will not keep a reference to the data buffer
                          after the functor returns.

        The delivery functor is called as
            fn(data)

        FUTURE: a new argument partial_ok may be set to True if it is ok to
        call the delivery functor with less data than requested, and to keep
        calling it until all data has been delivered. The signature of the
        delivery functor gets changed to fn(data, offset, size). Offset is the
        absolute file offset. I.e. not relative to the requested offset.
        Passing partial_ok=True might elide some buffer copies if the
        caller is doing something simple (such as reading an uncompressed
        brick) where partial copies are possible, and the backend is in the
        cloud, and a longer lived cache is being maintained, and the cache
        block size is smaller than the requested size. That is a lot of ifs.
        There was some code to handle partial_ok but it has been removed.
        Get it from the git history if you really want it.
        """
        raise NotImplementedError

    @staticmethod
    def _nice(n):
        """Human readable number."""
        if n >= 1024*1024 and (n % (1024*1024)) == 0:
            return str(n//(1024*1024)) + " MB" # whole number of MB
        elif n >= 256*1024 and (n % (256*1024)) == 0:
            return str(n/(1024*1024)) + " MB" # e.g. 42.75 NB
        elif n >= 1024 and (n % 1024) == 0:
            return str(n/1024) + " kB"
        else:
            return str(n) + " bytes"

    def _validate_read(self, offset, size):
        if self._mode not in ("rb", "w+b", "r+b"):
            raise ZgyUserError("The file is not open for reading.")
        if offset is None or offset < 0:
            raise ZgyUserError("Invalid offset {0} {1} for reading.".format(
                str(type(offset)), str(offset)))
        if size is None or size < 1:
            raise ZgyUserError("Invalid size {0} {1} for reading.".format(
                str(type(size)), str(size)))
        # Beware of mixing python and numpy scalars on Windows.
        # If offset fits in np.int32 then this is what it gets
        # cast to, which could make the sum overflow. On Linux
        # with a slightly older Python but same numpy version
        # the sum always ends up as np.int64. However, in this
        # particular case the exception should never occur so
        # the consequence was only less robust code.
        if int(offset) + int(size) > self.xx_eof:
            # Might be an internal error or a corrupted file,
            # but let's report only the immediate error and not
            # try to guess.
            raise ZgyEndOfFile("Offset {0} size {1} is past EOF at {2}".format(
                self._nice(offset), size, self._nice(self.xx_eof)))

    def _validate_write(self, data, offset):
        if self._mode not in ("w+b", "r+b"):
            raise ZgyUserError("The file is not open for writing.")
        if offset is None or offset < 0:
            raise ZgyUserError("Invalid offset for writing.")
        if data is None or len(data) < 1:
            raise ZgyUserError("Invalid size for writing.")

    def _validate_readv(self, requests):
        for offset, size, delivery in requests:
            self._validate_read(offset, size)

    def _check_short_read(self, offset, size, got):
        """
        Throw a descriptive error if there was something wrong with the read.
        Currently works for local files only.
        """
        # TODO-Low, can I get this to work also for seismic store?
        # Either make a virtual _get_real_size(), or admit this is
        # local-file spacific and move it down to class LocalFile.

        # Work around issue with mixing numpy and Python ints.
        offset, size, got = (int(offset), int(size), int(got))
        if got == size:
            return
        msg = "Cannot read offset {0} size {1}: ".format(
            self._nice(offset), self._nice(size))
        if got > size:
            raise ZgyInternalError(msg + "got too much data: {0}.".format(
                self._nice(got)))
        elif offset + size > self.xx_eof:
            # This can only happen if I (bug!) forgot to call _validate_read.
            raise ZgyEndOfFile(msg + "past EOF at {2}.".format(
                self._nice(self.xx_eof)))
        elif os.stat(self._file.fileno()).st_size < self.xx_eof:
            # This can happen if opening /dev/null for read/write,
            # or if a write failed due to a full disk (and was not checked),
            # or I somehow (bug!) failed to keep track of eof while writing.
            # Or maybe some other process truncated the file.
            raise ZgyEndOfFile(msg + "File is shorter than expected: {0} / {1}.".format(
                self._nice(os.stat(self._file.fileno()).st_size),
                self._nice(self.xx_eof)))
        else:
            # The os returned a short read for no apparent reason.
            # Maybe the file is a special device other than /dev/null.
            raise ZgyEndOfFile(msg + "short read for unknown reason.")

    @staticmethod
    def _consolidate_requests(requests, *,
                              max_hole = 2*1024*1024,
                              max_size = 64*1024*1024,
                              force_align = None,
                              consolidate_overlaps = False,
                              eof = None):
        """
        Given a list of requests as passed to xx_readv, try to reduce
        the number of requests by consolidating adjacent or nearly
        adjacent reads. If successful this means we will be reading
        with larger block sizes.

        Return a new list of requests that the caller may pass on
        to xx_readv instead of the original.

        Remember that the callback functions specified with the
        original requests need to be called with the exact data
        that they expected. This means that in the consolidated
        list the callback functions need to be wrappers.

        Parameters:
            max_hole:    See class SDConfig for a description.
            max_size:    See class SDConfig for a description.
            force_align: See class SDConfig for a description.
            consolidate_overlaps: Set to True if you expect some of the
                      individual requests to overlap, and you are ok with
                      risking some corner cases. For example, if you
                      request a mutable buffer then the overlapping area
                      will be delivered to more than one recipient and
                      the buffer may or may not be shared between the two.
                      The default is False which causes the code to not
                      attempt consolidation of these. Less efficient
                      but also less surprises. In practice there should
                      never be any overlap anyway.
        """
        class ConsolidatedDelivery:
            """
            Helper function to distribute a single delivery from
            a consolidated read request to all the requesters.
            Slice the data so each requester gets exactly what
            they originally asked for.
            Note that if the delivered data is bytes or bytearray
            the slicing will create a copy. If it is a numpy array
            the slice will just return a more efficient view.
            Should I perhaps create a numpy array here?
            Note that if the original request had overlapping reads
            we might want to force a copy anyway. Because we don't
            know whether the recipient asked for a mutable buffer.
            It is tempting to disallow overlapping reads completely.
            Caveat: Handling reads past EOF may be tricky.
            I need some specific unit tests for that.
            """
            def __init__(self, group, begin):
                self._group = group
                self._begin = begin
            def __call__(self, data):
                for offset, size, delivery in self._group:
                    if delivery:
                        end = min(offset + size - self._begin, len(data))
                        beg = min(offset - self._begin, end)
                        delivery(data[beg:end])

        def _groupsize(g, force_align, *, eof):
            """
            Given a list of (offset, size, functor)
            return offset and size for the entire group.
            The offset is the linear offset from the start of the file;
            it has not yet been converted to segment and local offset.
            The returned value includes any padding for force_align.

            TODO-High the padding is WRONG, because the alignment should be
            done per segment. We may end up crossing segment boundaries
            needlessly. And/or going past EOF. Going past EOF is critical
            because in the subsequent call to _split_by_segment() we will
            end up trying to actually read that part.

            Crossing segment boundaries is less of a problem.

              - It will not happen if the headers are aligned at least to
                force_align, which is typically the cache bricksize.

              - It will not happen if the file was uploaded with sdutil.
                In that case there will be just one segment.

              - It is (alomst) not an issue if a proper cache is attached.

              - A naive cache can align to 256 KB, this virtually guarantees
                the header area will be sufficiently aligned if the file
                was initially stored on the cloud.

              - In other cases there will be a small performance penalty but
                only when reading close to a segment boundary or when reading
                the headers. Opening a file may see a noticeable slowdown
                but not I think anything dramatic.
            """
            beg = min([e[0] for e in g])
            end = max([e[0] + e[1] for e in g])
            assert beg == g[0][0]
            #assert end == g[-1][0] + g[-1][1] # might fail if overlap.
            if force_align:
                beg = (beg // force_align) * force_align
                end = ((end + force_align - 1) // force_align) * force_align
                if eof: end = min(end, eof)
            return beg, end - beg

        def _split_requests(requests, force_align, *, eof):
            """
            Make a list of lists, grouping requests that should be read
            in a single operation. Operates on linear addresses, so if
            any of the input requests crossed a segment boundary then
            this will also be the case for the output.
            """
            # TODO-Low: In the Python code some parameters are inherited from
            # calling method; this is confusing and wasn't actually intended.
            all_requests = []
            prev_request = (0, 0, None)
            for request in sorted(requests):
                hole = request[0] - (prev_request[0] + prev_request[1])
                if not all_requests:
                    all_requests = [[request]]
                elif (hole <= max_hole and
                      (consolidate_overlaps or hole >= 0) and
                      (not max_size or _groupsize(all_requests[-1] + [request], force_align, eof=eof)[1] <= max_size)):
                    all_requests[-1].append(request)
                else:
                    all_requests.append([request])
                prev_request = request
            return all_requests

        def _join_requests(all_requests, force_align, *, eof):
            """Create the final result containing one entry for each
            consolidated group."""
            new_requests = []
            for group in all_requests:
                # Short cut, probably not worth the trouble.
                #if len(group)==1 and not force_align: new_requests.append(group[0])
                offset, size = _groupsize(group, force_align, eof=eof)
                new_requests.append((offset, size, ConsolidatedDelivery(group, offset)))
            return new_requests

        def _print_requests(all_requests, name = "Requests:"):
            """For debugging only, print a list of list of requests."""
            if len(all_requests) == 0 or (len(all_requests) == 1 and len(all_requests[0]) == 0):
                print("    (empty)")
                return
            print(name)
            for group in all_requests:
                if len(all_requests) > 1: print("  Group:")
                prev_offset, prev_size = (None, None)
                msg = "    {0} offset {1:8X} end {2:8X} size {3:6X}"
                for offset, size, delivery in group:
                    if prev_offset is not None:
                        skip_offset = prev_offset + prev_size
                        skip_size = offset - (prev_offset + prev_size)
                        if skip_size != 0:
                            print(msg.format("skip", skip_offset, offset, skip_size))
                    print(msg.format("read", offset, offset+size, size))
                    prev_offset, prev_size = (offset, size)


        # main part of _consolidate_requests().
        all_requests = _split_requests(requests, force_align, eof=eof)
        new_requests = _join_requests(all_requests, force_align, eof=eof)
        if False and len(requests) != len(new_requests):
            print("Consolidated", len(requests), "into", len(new_requests))
            print("Requests:")
            _print_requests([requests])
            print("Consolidated:")
            _print_requests([new_requests], name="Consolidated:")
        old_size = sum([x[1] for x in requests])
        new_size = sum([x[1] for x in new_requests])
        assert new_size >= old_size
        return new_requests

    @property
    def threadsafe(self):
        return False

    @property
    def xx_iscloud(self):
        return False

class LocalFile(FileADT):
    def __init__(self, filename, mode, iocontext):
        super().__init__(filename, mode, iocontext)
        self._config = FileConfig(iocontext)
        self._file = open(filename, mode) if mode != "d" else None
        self._mode = mode
        self._name = filename
        self._eof = 0 if mode in ("wb", "w+b") else os.stat(filename).st_size

    @property
    def xx_eof(self):
        return self._eof

    def xx_close(self):
        mode = self._mode
        self._mode = None
        if mode and self._name:
            if mode == "d":
                with suppress(FileNotFoundError):
                    os.remove(self._name)
            else:
                self._file.close()
        self._file = None
        self._name = None

class LocalFileOther(LocalFile):
    def xx_read(self, offset, size, *, usagehint=UsageHint.Unknown):
        self._validate_read(offset, size)
        self._file.seek(offset, 2 if offset < 0 else 0)
        data = self._file.read(size)
        self._check_short_read(offset, size, len(data))
        return data

    def xx_readv(self, requests, *, parallel_ok=False, immutable_ok=False, transient_ok=False, usagehint=UsageHint.Unknown):
        for offset, size, delivery in requests:
            delivery(LocalFileOther.xx_read(self, offset, size, usagehint=usagehint))

    def xx_write(self, data, offset, *, usagehint=UsageHint.Unknown):
        self._validate_write(data, offset)
        self._file.seek(offset)
        self._eof = max(self._eof, offset + len(data))
        nbytes = self._file.write(data)
        if nbytes != len(data):
            raise ZgyInternalError("Short write to local file")
        return len(data) # Most callers ignore this anyway.

    @property
    def threadsafe(self):
        return False

class LocalFileLinux(LocalFile):
    def xx_read(self, offset, size, *, usagehint=UsageHint.Unknown):
        self._validate_read(offset, size)
        data = os.pread(self._file.fileno(), size, offset)
        self._check_short_read(offset, size, len(data))
        return data

    def xx_readv(self, requests, *, parallel_ok=False, immutable_ok=False, transient_ok=False, usagehint=UsageHint.Unknown):
        for offset, size, delivery in requests:
            delivery(LocalFileLinux.xx_read(self, offset, size, usagehint=usagehint))

    def xx_write(self, data, offset, *, usagehint=UsageHint.Unknown):
        self._validate_write(data, offset)
        self._eof = max(self._eof, offset + len(data))
        nbytes = os.pwrite(self._file.fileno(), data, offset)
        if nbytes != len(data):
            raise ZgyInternalError("Short write to local file")

    @property
    def threadsafe(self):
        return True

class SeismicStoreFile(FileADT):
    """
    Access data in seismic store as a linear file even when the dataset
    has multiple segments. There are some limitations on write.

      * Writes starting at EOF are allowed, and will cause a new segment
        to be written.

      * Writes starting past EOF, signifying a hole in the data, are not
       allowed.

      * Writes starting before EOF are only allowed if offset,size exactly
        matches a previous write. This will cause that segment to be rewritten.

      * Possible future extension: For the  last segment only offset
        needs to match. This means the last segment may be resized.

    For read the class provides a readv() method to do scatter/gather reads.
    The code will then consolidate adjacent bricks to get larger brick size
    sent to SDAPI. Optionally parallelize requests that cannot be consolidated.
    """
    def __init__(self, filename, mode, iocontext):
        if sd is None:
            raise ZgyMissingFeature("Seismic Store is not available")
        super().__init__(filename, mode, iocontext)
        self._config = SDConfig(iocontext)
        #print(self._config.dump())

        sdcred = (self._config.sdurl, self._config.sdapikey, self._config.sdtoken)
        if not all(sdcred):
            raise ZgyUserError("Missing credentials:" +
                               ("" if sdcred[0] else " $OPENZGY_SDURL") +
                               ("" if sdcred[1] else " $OPENZGY_SDAPIKEY") +
                               ("" if sdcred[2] else " $OPENZGY_TOKEN"))
        if mode in ("rb"):
            self._accessor = sd.SdReader(filename, sdcred)
            # Get the size of each segment. For efficiency assume that all
            # segments except the first and last will have the same size.
            # TODO-Medium: If sizes() is cheap just request all of them.
            numseg = self._accessor.count()
            if numseg <= 3:
                self._sizes = list(self._accessor.sizes(*range(numseg)))
            else:
                tmp = self._accessor.sizes(0, 1, numseg-1)
                self._sizes = [tmp[0]] + (numseg-2) * [tmp[1]] + [tmp[2]]
        elif mode in ("w+b"):
            #print(self._config.dump(), sep="\n")
            # Create new, deleting or truncating existing.
            self._accessor = sd.SdWriter(filename, sdcred, False, self._config.extra)
            # TODO-Medium: If the file existed already, the mutable parts of the
            # metadata is allowed to change. Data can only be written if
            # the file was completely empty i.e. with just segment 0.
            self._sizes = []
        elif False and mode in ("r+b"):
            # TODO-High open without truncating not supported yet.
            # Limited support would be to open a file with only headers.
            # Full support is much trickier, need to re-open last segment
            # and also preferably do an incremental update of statistics
            # and lowres bricks.
            # Also, legaltag (and maybe seismicmeta) might be ignored here.
            self._accessor = sd.SdWriter(filename, sdcred, True, self._config.extra)
            numseg = self._accessor.count()
            if numseg <= 3:
                self._sizes = list(self._accessor.sizes(*range(numseg)))
            else:
                tmp = self._accessor.sizes(0, 1, numseg-1)
                self._sizes = [tmp[0]] + (numseg-2) * [tmp[1]] + [tmp[2]]
        elif mode in ("d"):
            # TODO-Performance keep the SdUtil instance alive.
            # There is a cost creating a new one, especially if the
            # token needs a refresh.
            self._accessor = None
            try:
                with sd.SdUtil(sdcred) as u:
                    u.delete(filename)
            except RuntimeError as ex:
                # File not found is ignored. Other errors are reported back.
                mode = None
                if str(ex).find("does not exist") < 0:
                    raise
        else:
            raise ZgyUserError("Opening ZGY as " + mode + " is not supported.")
        self._mode = mode
        self._cached_data = None

    @property
    def xx_eof(self):
        return sum(self._sizes)

    def xx_close(self):
        mode = self._mode
        self._mode = None
        if self._accessor and mode and mode != "d":
            self._accessor.close()
        self._accessor = None

    def _split_by_segment(self, requests):
        """
        Given one or more (offset, size, ...) tuples, convert these
        to (segment_number, offset_in_seg, size_in_seg, outpos).

        "outpos" is the offset to store the data that was read, if
        it is to be stored sequentially in one large buffer.

        Request for data past EOF is converted to a request for data
        in the last segment plus one. Trying to read that segment from
        seismic store will fail. Note that if _sizes is changed to
        include the open segment at the end then this special handling
        makes not much difference. At least not if the argument has
        already been checked to not cross the real EOF.

        Note that the callers currently check for reads past EOF
        and raises an exception in that case. So for now the above
        paragraph if of academic interest only.

        The returned list might be longer than the input if any of the
        input requests crossed segment boundaries.
        The return might be shorter than the input or even empty if
        any input request was for 0 bytes [or data past EOF... NOT]

        The algorithm is O(n^2) on segment_count * request_count
        but both numbers should be small. If this actually becomes
        a problem then use binary search in self._cumsize to find
        the starting segment.

        Maybe simplify: This logic could be moved inside SDAPI or the
        SDAPI wrapper. Reads from segment "-1" imply linear access.
        There would be a slight change in that requests split due to
        crossing a segment boundary would not be parallelized. But
        that is expected to be a very rare occurrence. Caveat, be
        careful what happens if reading past EOF. The code can currently
        handle that by returning data from the "open segment". That logic
        might not be possible to put into SDAPI. So this is probably
        not a good idea.
        """
        result = []
        outpos = 0
        for entry in requests:
            offset, size = entry[:2]
            assert offset >= 0
            size = entry[1]
            seg = 0
            for segsize in self._sizes:
                this_size = max(0, min(size, segsize - offset))
                if this_size > 0:
                    result.append((seg, offset, this_size, outpos))
                    offset += this_size
                    outpos += this_size
                    size   -= this_size
                    # If not crossing segment boundary, size will be 0
                    # Otherwise offset will be segsize, meaning that in
                    # the next iteration offset will be 0 and size will
                    # be the remaining data to be read.
                seg += 1
                offset -= segsize
                if size <= 0: break
        if size > 0:
            # Data past EOF treated as if it were in segment+1.
            result.append((seg, offset, size, outpos))
        insize = sum([e[1] for e in requests])
        outsize = result[-1][2] + result[-1][3] if result else 0
        assert insize == outsize
        return result

    def _cached_read(self, seg, offset, view):
        """
        Very simplistic cache implementation. Only store the most recent
        read from seismic store, and only consider a match when the range
        matches exactly. Also, always make copies both when copying in to
        and out of the cache. I.e. ignore immutable_ok, transient_ok.

        The cache may be useful if the upstream caller only asks for a
        single block at a time, so we get neither consolidation nor
        parallel access. Enabling this cache and setting force_align
        to a suitable value will hopefully cause the code to see the
        same bulk request happen more than once. If force_align is off
        it is very unlikely that the cache will help.

        Understand that storing just a single block in the cache will
        only help in lab conditions or in applications that we know
        for sure will issue requests for sequential parts of the cube.
        And if we know this already then we ought to be able to modify
        that code to pass down larger requests. Bottom line, it isn't
        very useful the way it works today.
        """
        if not self._config.aligned:
            self._config._debug_trace("read", len(view), len(view), 1)
            return self._accessor.read(seg, offset, view)
        seg_size = self._sizes[seg]
        a_beg = (offset // self._config.aligned) * self._config.aligned
        a_end = ((offset + len(view) + self._config.aligned - 1) // self._config.aligned) * self._config.aligned
        a_end = min(a_end, seg_size)
        c_seg, c_beg, c_data = self._cached_data or (0, 0, bytes())
        need = (seg, a_beg, a_end)
        have = (c_seg, c_beg, c_beg + len(c_data))
        #print('cache', need, ('==' if need == have else '<>'), have)
        if need == have:
            self._config._debug_trace("cachehit", len(view), a_end-a_beg, 1)
            data = c_data
        else:
            self._config._debug_trace("cachemiss", len(view), a_end-a_beg, 1)
            data = bytearray(a_end - a_beg)
            self._accessor.read(seg, a_beg, data)
            self._cached_data = (seg, a_beg, data)
        view[:] = data[offset-a_beg:offset-a_beg+len(view)]

    def xx_read(self, in_offset, in_size, *, usagehint=UsageHint.Unknown):
        self._validate_read(in_offset, in_size)
        work     = self._split_by_segment([(in_offset, in_size)])
        result   = bytearray(in_size)
        view     = memoryview(result)
        maxseg = max([seg for seg, offset, size, outpos in work])
        if maxseg >= len(self._sizes):
            # This should only happen in white box unit tests.
            # The higher levels of ZGY should have checked for EOF already.
            # But seismic store can return a really obscure error message
            # and/or hang in a retry loop if we don't do this check.
            raise ZgyEndOfFile("Attempt to read from segment " + str(maxseg))
        for seg, offset, size, outpos in work:
            self._cached_read(seg, offset, view[outpos:outpos+size])
        return result

    def xx_readv(self, requests, *, parallel_ok=False, immutable_ok=False, transient_ok=False, usagehint=UsageHint.Unknown):
        """
        Handle both brick consolidation and multi threading.

        This implementation will issue a single readv() request to the
        seismic store wrapper, wait for all threads to complete, and
        then deliver all the results. For this reason it needs to
        allocate a buffer to hold the entire data to be read.

        In the future it might be possible to have the seismic store
        wrapper support delivery callbacks and for it to allocate
        the result buffers itself. This saves some memory and also
        allows data to be decompressed if needed and copied out to
        user space as the bricks become available. Caveat: requests
        may need to be split if they cross a segment boundary.
        This means that we need support for partial delivery.
        Which would complicate things a lot.
        """
        self._validate_readv(requests)

        # I don't really like this kind of short cut since it creates
        # a lot of corner cases to test for. But, if the naive caching
        # is in effect then it is needed to make caching work.
        # If multiple requests then the cache won't work anyway,
        # and we might as well clear any data it contains.
        # TODO-Performance, can I move this test after consolidate
        # and split? Otherwise it will probably only work for the headers
        # and when the application really did fetch just one brick at
        # time. It might be good enough for Petrel though.
        if self._config.aligned and len(requests) == 1:
            for offset, size, delivery in requests:
                delivery(SeismicStoreFile.xx_read(self, offset, size, usagehint=usagehint))
            return
        self._cached_data = None

        # For debugging / logging only
        asked = sum([e[1] for e in requests])

        new_requests = self._consolidate_requests(requests,
                                                  max_hole=self._config.maxhole,
                                                  max_size=self._config.maxsize,
                                                  force_align=self._config.aligned,
                                                  eof=self.xx_eof)
        work     = self._split_by_segment(new_requests)
        # TODO-Low: For robustness scan work[] to get realize. As the
        # C++ code in impl/file_sd.cpp SeismicStoreFile::xx_readv() does.
        realsize = work[-1][2] + work[-1][3] if work else 0
        data     = bytearray(realsize)
        view     = memoryview(data)
        eof      = sum(self._sizes)

        # Read bulk data from seismic store using multiple threads.
        self._config._debug_trace("readv", asked, len(view), len(work))
        self._accessor.readv(work, data, self._config.threads)

        # Deliver result to caller.
        pos = 0
        for offset, size, delivery in new_requests:
            size = max(0, min(size, eof - offset))
            delivery(view[pos:pos+size])
            pos += size

    def xx_write(self, data, offset, *, usagehint=UsageHint.Unknown):
        self._validate_write(data, offset)
        current_eof = SeismicStoreFile.xx_eof.__get__(self) # nonvirtual call
        #print("SeismicStoreFile.xx_write(offset={0}, size={1}, current EOF is {2}".format(offset, len(data), current_eof))
        if offset == current_eof:
            # Sequential write from current EOF.
            # Specific limitation for ZGY, for performance reasons only.
            # This means we don't need to fetch sizes for all segments
            # when opening a file for read. Note that since the last
            # segment can have any size we won't discover a violation
            # until the following read.
            if len(self._sizes) >= 3 and self._sizes[-1] != self._sizes[1]:
                raise ZgyUserError("Cannot write arbitrarily sized segment.")
            self._config._debug_trace("append", len(data), len(data), 1)
            self._accessor.write(len(self._sizes), data, False)
            self._sizes.append(len(data))
        elif offset < current_eof:
            # Rewrite existing block. Resizing not allowed.
            seg = 0
            for segsize in self._sizes:
                if offset == 0:
                    if len(data) == segsize:
                        self._config._debug_trace("write", len(data), len(data), 1)
                        self._accessor.write(seg, data, True)
                        break
                    else:
                        raise ZgySegmentIsClosed("Cannot write resized segment.")
                elif offset < segsize:
                    raise ZgySegmentIsClosed("Cannot write part of segment.")
                seg += 1
                offset -= segsize
        else:
            # Attempting to write sparse data.
            raise ZgyUserError("Cannot write segments out of order.")
        return len(data)

    # If I want to disable threading, possibly also consolidation:
    #xx_readv = FileADT._forward_consolidated_readv
    #xx_readv = FileADT._forward_readv

    @property
    def threadsafe(self):
        return True if self._mode in ("rb") else False

    @property
    def xx_iscloud(self):
        return True

class SeismicStoreFileDelayedWrite(FileADT):
    """
    Improve on SeismicStoreFile, have it buffer large chunks of data before
    writing it out to a new segment.

      * Writes starting at EOF are allowed, and will buffer data in the
       "open segment" until explicitly flushed.

      * Writes starting past EOF, signifying a hole in the data, are not
       allowed.

      * Writes fully inside the open segment are allowed.

      * Writes starting before the open segment are only allowed if
        offset,size exactly matches a previous write. This will cause that
        segment to be rewritten. As a corollary, writes canot span the
        closed segment / open segment boundary.

      * Possible future extension: For the  last segment only offset
        needs to match. This means the last segment may be resized.
        Why we might want this: On opening a file with existing
        data bricks we might choose to read the last segment and
        turn it into an open segment. Then delete (in memory only)
        the last segment. When it is time to flush the data it gets
        rewritten. This allows adding bricks to a file, while still
        ensuring that all segments except first and last need to be
        the same size. Note that there are other tasks such as
        incrementally updating statistics and histogram that might
        turn out to be a lot of work.

      * When used to create ZGY files, caller must honor the convention
        that all segments except the first and last must have the same size.

      * Caveat: The fact that random writes are sometimes allowed, sometimes
        not depending on the segment number violates the principle of
        least surprise. And makes for more elaborate testing. For ZGY
        it is quite useful though. ZGY can recover from a ZgySegmentIsClosed
        exception by abandoning (leaking) the current block and write it
        to a new location. With a typical access pattern this will happen
        only occasionally.
    """
    def __init__(self, filename, mode, iocontext):
        super().__init__(filename, mode, iocontext)
        self._relay = SeismicStoreFile(filename, mode, iocontext)
        self._mode = mode
        self._open_segment = bytearray()
        self._usage_hint = None
        self._config = self._relay._config

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.xx_close()

    def _flush_part(self, this_segsize):
        """
        Flush "this_segsize" pending bytes. Leave any residual data
        in the open segment buffer.
        """
        assert this_segsize <= len(self._open_segment)
        assert len(self._open_segment) > 0
        flushme = memoryview(self._open_segment)
        nbytes = self._relay.xx_write(flushme[:this_segsize],
                                      self._relay.xx_eof,
                                      usagehint=self._usage_hint)
        if nbytes != this_segsize:
            raise ZgyInternalError("Short write to seismic store")
        self._open_segment = bytearray(flushme[this_segsize:])

    def _flush(self, final):
        """
        Flush pending writes, but only if we have enough data to fill
        one or more complete segments or if the file is being closed.
        The last segment is allowed to be smaller than the others.
        """
        if self._config.segsize > 0:
            while len(self._open_segment) >= self._config.segsize:
                self._flush_part(self._config.segsize)
        if final and len(self._open_segment) > 0:
            self._flush_part(len(self._open_segment))
        if len(self._open_segment) == 0:
            self._usage_hint = None

    @property
    def xx_eof(self):
        """
        Current size of the zgy file, including any buffered unwritten data.
        """
        return self._relay.xx_eof + len(self._open_segment)

    def xx_write(self, data, offset, *, usagehint=UsageHint.Unknown):
        """
        Write data to seismic store, buffering the writes to get larger
        segment sizes. Writes are only allowed at offset 0 and at EOF.
        This is less general then the parent type which lets us rewrite
        any segment as long as its size does not change.

        Segment 0 contains just the headers and is always written in one
        operation, so this is not buffered. Segment 0 can be both smaller
        and larger than segsize. Which is another reason to bypass the
        buffering code. Also, if we are rewriting data we bypass the
        buffering and require that the caller updates the entire segment.
        ZGY will currently only rewrite segment 0.

        If segsize is zero no buffering is done and each write will either
        create a new segment or completely rewrite an existing segment.
        """
        written = self.xx_eof
        committed = self._relay.xx_eof
        #print("SeismicStoreFileDelayedWrite.xx_write(offset={0}, size={1}, current EOF is {2}".format(offset, len(data), self.xx_eof))
        # TODO-Low: Refactor: Technically I could buffer segment 0 as well,
        # and leave it to the caller to immediately flush that segment.
        # I probably still need some special handling for the first segment.
        # The benefit is that FileADT gets less knowledge about ZGY proper.
        if offset == 0 or self._config.segsize <= 0 or offset < committed:
            return self._relay.xx_write(data, offset, usagehint=usagehint)
        if offset > written:
            # Write sparse data with a hole between written and offset.
            raise ZgyUserError("Cannot write segments out of order.")
        # TODO-Low: Generalize: If caller doesn't catch ZgySegmentIsClosed then all
        # rewrites ought to be forbidden. Since ZGY is our only client and
        # does in fact catch that exception then this is low priority.
        #if offset != self.xx_eof:
        #    raise ZgyUserError("Can only write at offset 0 or EOF, not {0} when EOF is {1}. Also, size is {2} and current open segment has {3} bytes.".format(offset, self.xx_eof, len(data), len(self._open_segment)))
        # TODO-Low: Refactor: Python abuse! Passing a "self" that is not an
        # instance of FileADT. But it has the attributes expected by the method.
        FileADT._validate_write(self, data, offset)
        lll = len(self._open_segment)
        if offset == written:
            # Append data to open segment
            self._open_segment += data
        elif offset + len(data) <= written:
            # Update data fully inside open segment
            self._open_segment[offset-committed:offset-committed+len(data)] = bytearray() + data
        else:
            # part update, part new.
            raise NotImplementedError() # TODO-Low support for symmetry.
        if self._usage_hint is None:
            self._usage_hint = usagehint
        elif self._usage_hint != usagehint:
            self._usage_hint = UsageHint.Unknown # mixed hints
        self._flush(False)
        return len(data) # TODO-Low: retval not useful if I throw on short reads.

    def xx_read(self, offset, size, *, usagehint=UsageHint.Unknown):
        FileADT._validate_read(self, offset, size)
        closed_size = max(0, min(size, self._relay.xx_eof - offset))
        opened_size = size - closed_size
        local_offset = max(0, offset - self._relay.xx_eof)
        if local_offset + opened_size > len(self._open_segment):
            raise ZgyUserError("Reading past EOF")
        data1 = self._relay.xx_read(offset, closed_size, usagehint=usagehint) if closed_size > 0 else None
        data2 = memoryview(self._open_segment)[local_offset:local_offset+opened_size] if opened_size > 0 else None
        return data1 + data2 if data1 and data2 else data1 or data2

    def xx_readv(self, requests, *, parallel_ok=False, immutable_ok=False, transient_ok=False, usagehint=UsageHint.Unknown, **kwargs):
        end = max([offset + size for offset, size, delivery in requests])
        if end <= self._relay.xx_eof:
            # The open segment is not involved, so just forward the request.
            self._relay.xx_readv(requests,
                                 parallel_ok=parallel_ok,
                                 immutable_ok=immutable_ok,
                                 transient_ok=transient_ok,
                                 usagehint=usagehint)
        else:
            # Let xx_read handle the requests one at a time.
            # If the requests consisted of both open and closed segments
            # then this is inefficient since SD access won't be paralellized.
            # But that case would be a lot of effort to support and it
            # won't happen often.
            for offset, size, delivery in requests:
                delivery(self.xx_read(offset, size, usagehint=usagehint))

    def xx_close(self):
        self._flush(True)
        return self._relay.xx_close()

    @property
    def threadsafe(self):
        return self._relay.threadsafe

    @property
    def xx_iscloud(self):
        return self._relay.xx_iscloud

def FileFactory(filename, mode, iocontext):
    """
    Return a FileADT instance able to read and/or write to the named file.
    In the future the function might return different types of instances
    e.g. if filename refers to something on the cloud.
    """
    if filename[:5] == "sd://":
        if mode in ("r+b", "w+b"):
            myfile = SeismicStoreFileDelayedWrite(filename, mode, iocontext)
        else:
            myfile = SeismicStoreFile(filename, mode, iocontext)
    elif hasattr(os, "pread"):
        myfile = LocalFileLinux(filename, mode, iocontext)
    else:
        myfile = LocalFileOther(filename, mode, iocontext)
    return myfile

# Copyright 2017-2021, Schlumberger
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
