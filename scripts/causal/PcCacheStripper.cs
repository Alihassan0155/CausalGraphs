// Streaming protocol-5 pickle transcode. Never unpickles CI values or retains CI keys.
// Intended ONLY for trusted local causal-learn FisherZ checkpoints. Fail closed on
// unsupported opcodes, unexpected cache layout or cross-references into discarded data.
using System;
using System.IO;
using System.Text;
using System.Collections.Generic;
using System.Diagnostics;

public sealed class PcStripResult {
    public long InputBytes, OutputBytes, InputMemos, OutputMemos, DroppedMemos, CIKeys;
    public long FastCIKeys;
    public int Caches;
    public double Seconds;
}

public static class PcCacheStripper {
    sealed class Reader : IDisposable {
        readonly FileStream file;
        readonly byte[] buffer = new byte[4*1024*1024];
        int position, length;
        public long Offset;
        public Reader(string path) { file = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, buffer.Length, FileOptions.SequentialScan); }
        public int Byte() {
            if (position == length) { length=file.Read(buffer,0,buffer.Length); position=0; if(length==0) throw new EndOfStreamException(); }
            Offset++; return buffer[position++];
        }
        public ulong Number(int n, Stream copy) {
            ulong value=0;
            for(int i=0;i<n;i++) { int b=Byte(); value|=(ulong)b<<(8*i); if(copy!=null)copy.WriteByte((byte)b); }
            return value;
        }
        public void Payload(ulong n, Stream copy) {
            while(n>0) {
                if(position==length) { length=file.Read(buffer,0,buffer.Length); position=0; if(length==0)throw new EndOfStreamException(); }
                int take=(int)Math.Min((ulong)(length-position),n);
                if(copy!=null)copy.Write(buffer,position,take);
                position+=take; Offset+=take; n-=(ulong)take;
            }
        }
        public byte[] Small(ulong n) {
            if(n>65536)throw new InvalidDataException("Unexpected large Unicode string");
            byte[] bytes=new byte[(int)n]; for(int i=0;i<bytes.Length;i++)bytes[i]=(byte)Byte(); return bytes;
        }
        bool MemoGet(ref int p,long memoCount) {
            long index;
            if(buffer[p]=='h') {index=buffer[p+1];p+=2;}
            else if(buffer[p]=='j') {index=(long)buffer[p+1]|((long)buffer[p+2]<<8)|((long)buffer[p+3]<<16)|((long)buffer[p+4]<<24);p+=5;}
            else return false;
            return index<memoCount;
        }
        public bool FastCI(ref long memoCount) {
            // Common NumPy float64-cache record. Look ahead WITHOUT consuming on
            // failure; frame boundaries and unusual encodings use the generic parser.
            if(length-position<300||buffer[position]!=0x8c)return false;
            int n=buffer[position+1],first=position+2,p=first;
            if(n==0||buffer[p]<'0'||buffer[p]>'9')return false;
            bool semicolon=false;
            for(int i=0;i<n;i++) {
                byte c=buffer[p++];if(c==';')semicolon=true;
                else if(!((c>='0'&&c<='9')||c=='|'||c=='.'))return false;
            }
            if(!semicolon||buffer[p++]!=0x94)return false; // key memo
            if(!MemoGet(ref p,memoCount+1)||!MemoGet(ref p,memoCount+1))return false;
            if(buffer[p++]!='C'||buffer[p++]!=8)return false;
            p+=8;
            if(buffer[p++]!=0x94||buffer[p++]!=0x86||buffer[p++]!=0x94||buffer[p++]!='R'||buffer[p++]!=0x94)return false;
            Offset+=p-position;position=p;memoCount+=4;return true;
        }
        public void Dispose(){file.Dispose();}
    }
    static void PutGet(Stream output, long index) {
        if(index<256) {output.WriteByte((byte)'h'); output.WriteByte((byte)index);}
        else {output.WriteByte((byte)'j'); for(int i=0;i<4;i++)output.WriteByte((byte)(index>>(8*i)));}
    }
    static byte[] Literal(byte opcode, byte[] prefix, byte[] value) {
        byte[] result=new byte[1+prefix.Length+value.Length]; result[0]=opcode;
        Buffer.BlockCopy(prefix,0,result,1,prefix.Length); Buffer.BlockCopy(value,0,result,1+prefix.Length,value.Length); return result;
    }
    static byte[] GlobalLiteral(string module,string name) {
        byte[] a=Encoding.UTF8.GetBytes(module), b=Encoding.UTF8.GetBytes(name);
        using(var s=new MemoryStream()) {
            s.WriteByte(0x8c);s.WriteByte((byte)a.Length);s.Write(a,0,a.Length);
            s.WriteByte(0x8c);s.WriteByte((byte)b.Length);s.Write(b,0,b.Length);s.WriteByte(0x93);
            return s.ToArray();
        }
    }
    static bool IsCIKey(byte[] value) {
        if(value.Length==0 || value[0]<'0' || value[0]>'9')return false;
        bool semicolon=false;
        for(int i=0;i<value.Length;i++) {
            byte c=value[i]; if(c==';')semicolon=true;
            else if(!((c>='0'&&c<='9')||c=='|'||c=='.'))return false;
        }
        return semicolon;
    }
    public static PcStripResult Rewrite(string input,string output) {
        if(Path.GetFullPath(input).Equals(Path.GetFullPath(output),StringComparison.OrdinalIgnoreCase))throw new IOException("Input and output must differ");
        var result=new PcStripResult(); var timer=Stopwatch.StartNew(); double nextReport=2;
        var memo=new Dictionary<long,long>();
        var strings=new Dictionary<long,string>();
        bool cache=false,expectCache=false,rootMemo=false,done=false;
        int markDepth=0;
        long sourceMemo=0,outputMemo=0;
        byte[] pending=null;
        string pendingString=null,lastString=null,previousString=null;
        using(var reader=new Reader(input))
        using(var file=new FileStream(output,FileMode.CreateNew,FileAccess.Write,FileShare.None,1024*1024)) {
            while(!done) {
                if(cache&&reader.FastCI(ref sourceMemo)) {
                    result.CIKeys++;result.FastCIKeys++;pending=null;pendingString=null;continue;
                }
                int op=reader.Byte();
                if(op==0x95) {reader.Number(8,null);continue;} // remove FRAME envelopes
                if(op==0x94) { // MEMOIZE: dense renumbering, not sparse multi-GB memo arrays
                    bool keep=!cache || rootMemo || pending!=null;
                    if(keep) {
                        if(cache&&!rootMemo)file.Write(pending,0,pending.Length);
                        file.WriteByte(0x94);
                        if(cache&&!rootMemo)file.WriteByte((byte)'0'); // keep reusable atom, discard stack value
                        memo.Add(sourceMemo,outputMemo++);
                        if(pendingString!=null)strings[sourceMemo]=pendingString;
                    }
                    sourceMemo++; rootMemo=false; pending=null; pendingString=null;continue;
                }
                pending=null; pendingString=null;
                if(op=='h'||op=='j') {
                    long index=(long)reader.Number(op=='h'?1:4,null);
                    if(!cache) {
                        long mapped;
                        if(!memo.TryGetValue(index,out mapped))throw new InvalidDataException("External reference into discarded cache memo "+index+" at byte "+reader.Offset);
                        PutGet(file,mapped);
                    }
                    string text;
                    if(strings.TryGetValue(index,out text)){previousString=lastString;lastString=text;}
                    continue;
                }
                if(op==0x8c||op=='X'||op==0x8d) {
                    int width=op==0x8c?1:op=='X'?4:8;
                    ulong n=reader.Number(width,null);
                    byte[] bytes=reader.Small(n);
                    bool ci=cache&&IsCIKey(bytes);
                    string text=ci?null:Encoding.UTF8.GetString(bytes);
                    if(cache&&markDepth==0&&(text=="method"||text=="correlation_matrix")) {
                        cache=false; // the cache object is complete; preserve next FisherZ state field
                    }
                    byte[] prefix=new byte[width];for(int i=0;i<width;i++)prefix[i]=(byte)(n>>(8*i));
                    if(!cache) {
                        file.WriteByte((byte)op);file.Write(prefix,0,prefix.Length);file.Write(bytes,0,bytes.Length);
                        if(text=="pvalue_cache") { if(result.Caches!=0)throw new InvalidDataException("Multiple cache fields unsupported");expectCache=true; }
                        pendingString=text;
                    } else if(ci) {
                        result.CIKeys++;
                    } else {
                        // A tiny set of metadata/module/name strings may be referenced after
                        // the cache. Retain these atoms, NEVER the millions of CI keys.
                        pending=Literal((byte)op,prefix,bytes);pendingString=text;
                    }
                    if(text!=null){previousString=lastString;lastString=text;}
                    continue;
                }
                if(expectCache) {
                    if(op!='}')throw new InvalidDataException("Cache field not followed by EMPTY_DICT");
                    file.WriteByte((byte)op);expectCache=false;cache=true;rootMemo=true;markDepth=0;result.Caches++;continue;
                }
                if(cache&&op==0x93) {
                    if((previousString=="numpy.core.multiarray"||previousString=="numpy._core.multiarray")&&lastString=="scalar")
                        pending=GlobalLiteral(previousString,lastString);
                    // Other opaque values are dropped. A later external reference fails closed.
                }
                if(cache) {
                    if(op=='(')markDepth++;
                    if(op=='t'||op=='l'||op=='d'||op=='e'||op=='u'||op=='1'||op==0x90||op==0x91) {
                        markDepth--; if(markDepth<0)throw new InvalidDataException("Unbalanced cache MARK");
                    }
                } else file.WriteByte((byte)op);
                Stream destination=cache?null:file;
                switch(op) {
                    case 0x80:
                        if(reader.Number(1,destination)!=5)throw new InvalidDataException("Only source protocol 5 supported");break;
                    case 'C': case 0x8a: {ulong n=reader.Number(1,destination);reader.Payload(n,destination);break;}
                    case 'B': case 0x8b: {ulong n=reader.Number(4,destination);reader.Payload(n,destination);break;}
                    case 0x8e: case 0x96: {ulong n=reader.Number(8,destination);reader.Payload(n,destination);break;}
                    case 'G':reader.Payload(8,destination);break;
                    case 'J':reader.Payload(4,destination);break;
                    case 'K':reader.Payload(1,destination);break;
                    case 'M':reader.Payload(2,destination);break;
                    case '.': if(cache||expectCache||result.Caches!=1)throw new InvalidDataException("Incomplete cache transcode");done=true;break;
                    case '(': case ')': case ']': case '}': case 'N': case 'R': case 'b':
                    case 'a': case 'e': case 's': case 'u': case 't': case 'l': case 'd':
                    case '0': case '1': case '2': case 0x81: case 0x85: case 0x86: case 0x87:
                    case 0x88: case 0x89: case 0x8f: case 0x90: case 0x91: case 0x92: case 0x93:break;
                    default:throw new InvalidDataException("Unsupported opcode 0x"+op.ToString("X2")+" at byte "+reader.Offset);
                }
                if(timer.Elapsed.TotalSeconds>=nextReport) {
                    Console.Error.WriteLine("strip_progress bytes="+reader.Offset+" ci_keys="+result.CIKeys+" retained_memos="+outputMemo+" seconds="+timer.Elapsed.TotalSeconds.ToString("F1"));
                    nextReport=timer.Elapsed.TotalSeconds+2;
                }
            }
            result.InputBytes=reader.Offset;file.Flush(true);result.OutputBytes=file.Length;
        }
        result.InputMemos=sourceMemo;result.OutputMemos=outputMemo;
        result.DroppedMemos=sourceMemo-outputMemo;result.Seconds=timer.Elapsed.TotalSeconds;
        return result;
    }
}
