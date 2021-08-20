#!/usr/bin/env python3

"""File: cppmeta.py
This massive kludge is/was used for the Python to C++ port,
giving a starting point and hopefully reducing the number
of copy/paste errors. It probably isn't much use after the
port is done, but I might keep it in case a really big
change needs to be done to the format.
WARNING: THE GENERATED CODE IS JUST A STARTING POINT.
"""

from ..impl import meta
from collections import namedtuple

_fixedType = namedtuple("Attribute", "cname dims ptype ctype csize comment atype aread")

def fixtuple(e):
    """
    from: ('_size',    '3i', 'int32[3]', '...')
    to:   ('_size[3]', '3i', 'int32', 3, '...')
    """
    cname = e[0]
    ptype = e[1]
    ctype = e[2]
    csize = 0
    dims = ""
    p1 = ctype.find("[")
    p2 = ctype.find("]")
    if p1 > 0 and p2 > p1:
        csize = ctype[p1+1:p2]
        dims = "[" + csize + "]"
        try:
            csize = int(csize)
        except ValueError:
            csize = -1
        #cname = cname + "[" + csize + "]"
        ctype = ctype[0:p1]
    try:
        ctype = {
            "float32": "float",
            "float64": "double",
            "uint8":   "std::uint8_t",
            "uint16":  "std::uint16_t",
            "uint32":  "std::uint32_t",
            "uint64":  "std::uint64_t",
            "int8":    "std::int8_t",
            "int16":   "std::int16_t",
            "int32":   "std::int32_t",
            "int64":   "std::int64_t",
        }[ctype]
    except KeyError:
        pass
    # Also suggest a fixed-length std::array type,
    # which might work better in access functions.
    if ctype == "char*":
        atype = "std::string"
        aread = 'std::string(_pod.{0} ? _pod.{0} : "")'.format(cname)
    elif csize:
        atype = "std::array<{0},{1}>".format(ctype, csize)
        aread = "ptr_to_array<{0},{1}>(_pod.{2})".format(ctype, csize, cname)
    else:
        atype = ctype
        aread = "align(_pod." + cname + ")"
    return _fixedType(cname, dims, ptype, ctype, csize, e[3], atype, aread)

def dumpformats(cls, *, file = None):
    file = file or sys.stdout
    say = lambda *args, **kwargs: print(*args, **kwargs, file=file)
    classname = cls.__name__
    basename = "I" + (classname[:-2] if classname[-2]=="V" else classname)+"Access"

    if classname[-2:] == "V1":
        say()
        say("/" * 77)
        say("///   " + classname[:-2] + "   " + "/" * (70 - len(classname)))
        say("/" * 77)
        say("""
class {0} : public IHeaderAccess
{{
public:
  static std::shared_ptr<{0}> factory(std::uint32_t version);
  // TO""""""DO: Add pure virtual functions, more or less matching the signatures
  // not of {1} but of the latest version. Then modify older versions
  // so they match the expected api. And while you are at it, move dump()
  // in the latest version to the base class and remove the others.
public:
}};""".format(basename, classname))

    fixedformats = list([fixtuple(ee) for ee in cls._formats()]) if hasattr(cls, "_formats") else []

    # ----- Physical layout: class FooV{n}

    say("\n// Python layout is:", cls._format())
    say("// Size in storage:", cls.headersize() if hasattr(cls, "headersize") else 0, "bytes")
    say("#pragma pack(1)\nclass {0}POD\n{{".format(classname))
    say("public:")
    for e in fixedformats:
        say("  {3}{0:17} {1:15} // {2}".format(e.ctype, e.cname + e.dims + ";", e.comment, ("" if e.ptype else "//")))
    say("};\n#pragma pack()")

    # ------ Read accessors class FooV{n}Access: aggregates FooV{n}
    #        and inherits FooBase. Note that no example FooBase
    #        is generated. Use the latest FooV{n} and replace all
    #        overridden methods with pure virtual methods.

    say("\nclass {0}Access : public {1}\n{{\npublic:\n  {0} _pod;".format(classname, basename))
    say("  virtual void read(const std::shared_ptr<FileADT> file, std::int64_t offset) override;")
    say("  virtual void byteswap() override;")
    say("  virtual void rawdump(std::ostream& out, const std::string& prefix = \"\") override;")
    say("  virtual void dump(std::ostream& out, const std::string& prefix = \"\") override;")
    say("public:")
    notimpl = 'throw OpenZGY::Errors::ZgyInternalError("Not implemented");'
    for e in fixedformats:
        sig1 ="virtual {0}".format(e.atype)
        sig ="  {0:34} {1}() const override".format(sig1, e.cname[1:])
        if e.ptype or e.atype == "std::string":
            say(sig + " {{ return {0.aread}; }}".format(e))
        else:
            say(sig + " { " + notimpl +  " }")
    say("};")

    # ----- FooV{n}Access::read() implementation.
    say("\nvoid\n{0}Access::read(const std::shared_ptr<FileADT>& file, std::int64_t offset, std::int64_t size)".format(classname))
    say("{\n  file->xx_read(&this->_pod, offset, sizeof(this->_pod));\n  byteswap();\n}")

    # ----- FooV{n}Access::byteswap() implementation.
    say("\nvoid\n{0}Access::byteswap()\n{{".format(classname))
    for e in fixedformats:
        if e.ptype and e.atype != "std::string" and e.ctype in (
                "float", "double",
                "std::int16_t", "std::int32_t", "std::int64_t",
                "std::uint16_t", "std::uint32_t", "std::uint64_t"):
            if e.csize == 0:
                say("  byteswapT(&_pod.{e.cname});".format(e=e))
            else:
                say("  byteswapT(&_pod.{e.cname}[0], {e.csize});".format(e=e))
        else:
            if e.ptype:
                say("  // byteswap not needed for {e.ctype} {e.cname}{e.dims} because of its type.".format(e=e))
            else:
                say("  // byteswap not needed for {e.ctype} {e.cname}{e.dims} because it is not stored.".format(e=e))
    say("}")

    # ----- Debugging: FooV{n}Access::rawdump() implementation.
    #       This uses the POD data members directly.

    say("\nvoid\n{0}Access::rawdump(std::ostream& out, const std::string& prefix)\n{{".format(classname))
    say('  out')
    for e in fixedformats:
        if e.ptype:
            say('    << prefix << "{0:20s}"'.format(e.cname+":"), end="")
        else:
            say('    //<< prefix << "{0:20s}"'.format(e.cname+":"), end="")
        if not e.csize:
            say(' << _pod.{0} << "\\n"'.format(e.cname))
        else:
            for i in range(e.csize):
                say(' << _pod.{0}[{1}] << " "'.format(e.cname, i), end="");
            say(' << "\\n"')
    say("  ;")
    say("}")

    # ----- Debugging: FooV{n}Access::dump() implementation.
    #       This uses the generated access methods.
    #       NOTE: Should probably move the highest version of
    #       FooV{n}Access::dump to the FooBase base class
    #       and remove the others.
    say("\nvoid\n{0}Access::dump(std::ostream& out, const std::string& prefix)\n{{".format(classname))
    say('  out')
    for e in fixedformats:
        say('    << prefix << "{0:20s}"'.format(e.cname[1:]+"():"), end="")
        if not e.ptype:
            say(' << "N/A\\n"')
        elif not e.csize:
            say(' << {0}() << "\\n"'.format(e.cname[1:]))
        else:
            say(' << array_to_string({0}()) << "\\n"'.format(e.cname[1:]))
    say("  ;")
    say("}")

    @classmethod
    def checkformats(cls, verbose = False, *, file = None):
        """
        This is the old class-dumper, moved from impl.meta because it is
        not production code. In fact, its functinality is superceded by
        dumpformats but might, like dumpformats, turn out to be useful
        at some point. Yagni doesn't apply here.

        Helper to compare the python definition of the header layout with
        the C++ version. If verbose is True output the entire definition
        as it would appear in a C++ header. With verbose False it only
        does the consistency check. This is cheap enough to be permanently
        enabled. Also check that the same attribute isn't listed twice.
        """
        file = file or sys.stdout
        mapping = {
            "char*":   "",
            "enum":    "B",
            "float32": "f",
            "float64": "d",
            "int32":   "i",
            "int64":   "q",
            "uint32":  "I",
            "uint64":  "Q",
            "uint8":   "B",
        }
        errors = 0
        seen = set()
        byteoffset = 0
        if verbose == 1:
            print("// Python layout is:", cls._format(), file=file)
            print("// Size in storage:", cls.headersize(), "bytes", file=file)
            print("class {0}\n{{".format(cls.__name__), file=file)
        if verbose == 2:
            print("<h3>{0}</h3>".format(cls.__name__), file=file)
            print('<table border="1" style="border-collapse: collapse">', file=file)
            print("<tr><th>offset</th><th>size</th><th>type</th><th>name</th><th>remarks</th></tr>", file=file)
        for e in cls._formats():
            if e[0] in seen:
                print("# ERROR: attribute {0} is listed twice.".format(e[0]), file=file)
            seen.add(e[0])
            ctype = e[2]
            cname = e[0]
            csize = None
            p1 = ctype.find("[")
            p2 = ctype.find("]")
            if p1 > 0 and p2 > p1:
                csize = ctype[p1+1:p2]
                cname = cname + "[" + csize + "]"
                ctype = ctype[0:p1]
            if verbose == 1:
                #print("  // offset", byteoffset, file=file)
                print("  {3}{0:10} {1:15} // {2}".format(ctype, cname + ";", e[3], ("" if e[1] else "//")), file=file)
            if verbose == 2:
                print("<tr><td>{3:3}</td><td>{4:2}</td><td>{0:10}</td><td>{1:15}</td><td>{2}</td></tr>".format(ctype, cname, e[3], byteoffset, struct.calcsize(e[1])), file=file)
            expect = (csize if csize else '') + mapping[ctype]
            if expect == "16B": expect = "16s"
            if expect == "4B": expect = "4s"
            actual = e[1]
            if actual and actual != expect:
                print("# ERROR: Expected code {0}, got {1}".format(expect, e[1]), file=file)
                errors += 1
            byteoffset += struct.calcsize(e[1])
        if verbose == 1:
            print("};", file=file)
        if verbose == 2:
            print('<tr><td>{0:3}</td><td colspan="3">&nbsp;</td><td>end</td></tr>'.format(byteoffset), file=file)
            print("</table>", file=file)
        assert not errors

def checkAllFormats(*, verbose = False, file = None):
    meta.FileHeader.checkformats(verbose=verbose, file=file)
    meta.InfoHeaderV1.checkformats(verbose=verbose, file=file)
    meta.InfoHeaderV2.checkformats(verbose=verbose, file=file)
    meta.InfoHeaderV3.checkformats(verbose=verbose, file=file)
    meta.HistHeaderV1.checkformats(verbose=verbose, file=file)
    meta.HistHeaderV2.checkformats(verbose=verbose, file=file)
    meta.HistHeaderV3.checkformats(verbose=verbose, file=file)

def dumpAllFormats(*, file = None):
    print("// AUTOGENERATED -- NEEDS MAJOR EDITS BEFORE USE\n", file=f)
    dumpformats(meta.FileHeader, file=f)
    dumpformats(meta.OffsetHeaderV1, file=f)
    dumpformats(meta.OffsetHeaderV2, file=f)
    dumpformats(meta.InfoHeaderV1, file=f)
    dumpformats(meta.InfoHeaderV2, file=f)
    dumpformats(meta.HistHeaderV1, file=f)
    dumpformats(meta.HistHeaderV2, file=f)

if __name__ == "__main__":
    # Consistency check only
    checkAllFormats()

    # Simple C++ header file (dumpformats does much more)
    #checkAllFormats(verbose=1)

    # HTML formatted documentation
    #checkAllFormats(verbose=2)

    with open("tmpmetatmp.h", "w") as f:
        dumpAllFormats(file = f)

# Copyright 2017-2020, Schlumberger
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
