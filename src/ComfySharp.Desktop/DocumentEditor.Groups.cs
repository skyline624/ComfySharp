using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Globalization;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Media;
using Avalonia.VisualTree;
using ComfySharp.Workflow;

namespace ComfySharp.Desktop;

public sealed partial class DocumentEditor
{
    private readonly ObservableCollection<GroupView> groupViews = [];
    private GroupDrag? groupDrag;
    private sealed record GroupDrag(int Index, Point Start, string Before, bool FrameOnly, IReadOnlyDictionary<NodeId, GraphRect> Bounds)
    {
        public Vector Delta { get; set; }
    }
    private Dictionary<NodeId, GraphRect> MeasuredNodeBounds()
    {
        var result = new Dictionary<NodeId, GraphRect>();
        foreach (var container in Canvas.GetVisualDescendants().OfType<Nodify.Avalonia.ItemContainer>())
            if (container.DataContext is NodeView view && container.Bounds.Width > 0 && container.Bounds.Height > 0)
                result[view.Id] = new(view.Location.X, view.Location.Y, container.Bounds.Width, container.Bounds.Height);
        return result;
    }
    private void ReloadGroups()
    {
        int selected = (GroupChoices.SelectedItem as GroupView)?.Index ?? -1; groupViews.Clear(); GroupDiagnostic.Text = "";
        try { foreach (var group in Document.Groups) groupViews.Add(new(group)); }
        catch (Exception error) when (error is FormatException or InvalidOperationException or ArgumentException)
        { GroupDiagnostic.Text = "Groups are preserved but cannot be displayed: " + error.Message; }
        GroupChoices.ItemsSource = groupViews;
        GroupChoices.SelectedItem = groupViews.FirstOrDefault(g => g.Index == selected) ?? groupViews.FirstOrDefault();
    }
    public int CreateGroupFromSelection()
    {
        int index = Document.GroupNodes(SelectedIds(), measuredBounds: MeasuredNodeBounds()); Reload();
        GroupChoices.SelectedItem = groupViews.Single(g => g.Index == index); return index;
    }
    public void MoveGroup(int index, double x, double y, bool frameOnly = false)
    {
        Document.MoveGroup(index, x, y, frameOnly, MeasuredNodeBounds()); Reload();
    }
    private int SelectedGroupIndex => (GroupChoices.SelectedItem as GroupView)?.Index ?? throw new InvalidOperationException("Select a group first.");
    private void CreateGroupClicked(object? sender, RoutedEventArgs e) => Try(() => CreateGroupFromSelection());
    private void GroupSelectionChanged(object? sender, SelectionChangedEventArgs e)
    {
        if (GroupChoices.SelectedItem is not GroupView group) return;
        GroupTitle.Text = group.Title; GroupColor.Text = group.Color;
        GroupWidth.Text = group.Width.ToString(CultureInfo.InvariantCulture); GroupHeight.Text = group.Height.ToString(CultureInfo.InvariantCulture);
    }
    private void ApplyGroupClicked(object? sender, RoutedEventArgs e) => Try(() =>
    {
        _ = Avalonia.Media.Color.Parse(GroupColor.Text ?? "#335");
        Document.UpdateGroup(SelectedGroupIndex, GroupTitle.Text ?? "Group", GroupColor.Text ?? "#335",
            double.Parse(GroupWidth.Text ?? "140", CultureInfo.InvariantCulture), double.Parse(GroupHeight.Text ?? "80", CultureInfo.InvariantCulture)); Reload();
    });
    private void PinGroupClicked(object? sender, RoutedEventArgs e) => Try(() => { int index = SelectedGroupIndex; Document.SetGroupPinned(index, !Document.Groups[index].Pinned); Reload(); });
    private void RemoveGroupClicked(object? sender, RoutedEventArgs e) => Try(() => { Document.RemoveGroup(SelectedGroupIndex); Reload(); });
    private Point GraphPoint(PointerEventArgs e)
    {
        var point = e.GetPosition(Canvas); return new(point.X / Canvas.ViewportZoom + Canvas.ViewportLocation.X, point.Y / Canvas.ViewportZoom + Canvas.ViewportLocation.Y);
    }
    private void GroupPointerPressed(object? sender, PointerPressedEventArgs e)
    {
        if (sender is not Control { DataContext: GroupView group } header || !e.GetCurrentPoint(header).Properties.IsLeftButtonPressed) return;
        GroupChoices.SelectedItem = group; e.Handled = true; if (group.Pinned) return;
        groupDrag = new(group.Index, GraphPoint(e), Document.ToJson(), e.KeyModifiers.HasFlag(KeyModifiers.Shift), MeasuredNodeBounds());
        e.Pointer.Capture(header);
    }
    private void GroupPointerMoved(object? sender, PointerEventArgs e)
    {
        if (groupDrag is not { } drag) return; e.Handled = true;
        try
        {
            if (Document.ToJson() != drag.Before) throw new InvalidOperationException("The document changed during group movement.");
            drag.Delta = GraphPoint(e) - drag.Start;
            var preview = WorkflowDocument.Parse(drag.Before); preview.MoveGroup(drag.Index, drag.Delta.X, drag.Delta.Y, drag.FrameOnly, drag.Bounds);
            foreach (var node in preview.Nodes) nodes.Single(v => v.Id == node.Id).PreviewLocation(new(node.X, node.Y));
            foreach (var group in preview.Groups) groupViews.Single(v => v.Index == group.Index).PreviewBounds(group.Bounds);
            RefreshConnections();
        }
        catch (Exception error) { groupDrag = null; e.Pointer.Capture(null); Reload(); Error?.Invoke(this, error.Message); }
    }
    private void GroupPointerReleased(object? sender, PointerReleasedEventArgs e)
    {
        if (groupDrag is not { } drag) return; groupDrag = null; e.Pointer.Capture(null); e.Handled = true;
        Try(() =>
        {
            try
            {
                if (Document.ToJson() != drag.Before) throw new InvalidOperationException("The document changed during group movement.");
                Document.MoveGroup(drag.Index, drag.Delta.X, drag.Delta.Y, drag.FrameOnly, drag.Bounds);
            }
            finally { Reload(); }
        });
    }
    private void GroupPointerCaptureLost(object? sender, PointerCaptureLostEventArgs e)
    {
        if (groupDrag is null) return; groupDrag = null; if (!disposed) Reload();
    }
}

public sealed class GroupView : INotifyPropertyChanged
{
    public int Index { get; }
    public string Title { get; }
    public string Color { get; }
    public bool Pinned { get; }
    public string Caption => (Pinned ? "[Pinned] " : "") + Title;
    public IBrush Brush { get; }
    public Point Location { get; private set; }
    public double Width { get; private set; }
    public double Height { get; private set; }
    public event PropertyChangedEventHandler? PropertyChanged;
    public GroupView(GraphGroup group)
    {
        Index = group.Index; Title = group.Title; Color = group.Color; Pinned = group.Pinned;
        Brush = Avalonia.Media.Color.TryParse(Color, out var color) ? new SolidColorBrush(color) : Brushes.SlateGray; PreviewBounds(group.Bounds);
    }
    internal void PreviewBounds(GraphRect bounds)
    {
        Location = new(bounds.X, bounds.Y); Width = bounds.Width; Height = bounds.Height;
        foreach (string property in new[] { nameof(Location), nameof(Width), nameof(Height) }) PropertyChanged?.Invoke(this, new(property));
    }
    public override string ToString() => Caption;
}
