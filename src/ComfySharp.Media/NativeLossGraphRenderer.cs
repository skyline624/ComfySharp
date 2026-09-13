using System.Globalization;
using System.Runtime.InteropServices;
using ComfySharp.Contracts;
using SkiaSharp;

namespace ComfySharp.Media;

public sealed record LossGraphPoint(int X,int Y);
public sealed record LossGraphLabel(int X,int Y,string Text);
public sealed record LossGraphLayout(IReadOnlyList<LossGraphPoint> Points,IReadOnlyList<LossGraphLabel> Labels);

/// <summary>Frozen LossGraphNode geometry, rendered with native Skia. Glyph rasterization
/// uses the platform font and is not a pixel-identical replacement for Pillow/FreeType.</summary>
public static class NativeLossGraphRenderer
{
    public const int Width=840,Height=520;
    public static LossGraphLayout Describe(IReadOnlyList<double> losses,CancellationToken cancellationToken=default)
    {
        ArgumentNullException.ThrowIfNull(losses);cancellationToken.ThrowIfCancellationRequested();
        if(losses.Count==0||losses.Any(v=>!double.IsFinite(v)))throw new ArgumentException("Loss graph requires a nonempty finite loss history.",nameof(losses));
        double min=losses.Min(),max=losses.Max(),range=max-min;
        if(range==0||!double.IsFinite(range))throw new ArithmeticException("The source loss graph cannot normalize a constant or overflowing loss range.");
        var points=new LossGraphPoint[losses.Count];
        for(int i=0;i<points.Length;i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            points[i]=new(40+(int)((double)i/losses.Count*800),480-(int)((losses[i]-min)/range*480));
        }
        // Python's formatting rounds to nearest-even; F2 uses the same binary floating-point value.
        return new(Array.AsReadOnly(points),Array.AsReadOnly(new[]{new LossGraphLabel(5,240,"Loss"),new LossGraphLabel(400,490,"Steps"),
            new LossGraphLabel(10,0,max.ToString("F2",CultureInfo.InvariantCulture)),new LossGraphLabel(10,470,min.ToString("F2",CultureInfo.InvariantCulture))}));
    }
    public static DecodedImageBatch Render(IReadOnlyList<double> losses,CancellationToken cancellationToken=default)
    {
        var layout=Describe(losses,cancellationToken);
        using var bitmap=new SKBitmap(new SKImageInfo(Width,Height,SKColorType.Rgba8888,SKAlphaType.Opaque));
        if(bitmap.GetPixels()==IntPtr.Zero)throw new OutOfMemoryException("Loss graph pixel allocation failed.");
        using var canvas=new SKCanvas(bitmap);canvas.Clear(SKColors.White);
        using var paint=new SKPaint{Color=SKColors.Blue,StrokeWidth=2,IsAntialias=false,StrokeCap=SKStrokeCap.Square};
        for(int i=1;i<layout.Points.Count;i++)
        {
            cancellationToken.ThrowIfCancellationRequested();var a=layout.Points[i-1];var b=layout.Points[i];
            canvas.DrawLine(a.X,a.Y,b.X,b.Y,paint);
        }
        paint.Color=SKColors.Black;canvas.DrawLine(40,0,40,480,paint);canvas.DrawLine(40,480,840,480,paint);
        using var typeface=SKTypeface.FromFamilyName("Arial");
        using var font=new SKFont(typeface??SKTypeface.Default,12);paint.IsAntialias=true;
        foreach(var label in layout.Labels)canvas.DrawText(label.Text,label.X,label.Y-font.Metrics.Ascent,font,paint);
        canvas.Flush();var rgb=new float[Width*Height*3];var row=new byte[Width*4];
        for(int y=0;y<Height;y++)
        {
            cancellationToken.ThrowIfCancellationRequested();Marshal.Copy(bitmap.GetPixels()+y*bitmap.RowBytes,row,0,row.Length);
            for(int x=0;x<Width;x++)for(int c=0;c<3;c++)rgb[(y*Width+x)*3+c]=row[x*4+c]/255f;
        }
        return new(Width,Height,1,rgb,null);
    }
}
