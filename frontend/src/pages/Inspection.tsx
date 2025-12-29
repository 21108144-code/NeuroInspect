import { useState, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { inspectImage } from '@/lib/api'
import { formatPercent, getSeverityClass, downloadBase64Image } from '@/lib/utils'
import type { InspectionResult, DefectDetection } from '@/lib/types'
import {
    Upload,
    Image as ImageIcon,
    AlertTriangle,
    CheckCircle,
    Download,
    RefreshCw,
    ZoomIn,
    Layers,
} from 'lucide-react'

export default function Inspection() {
    const [file, setFile] = useState<File | null>(null)
    const [preview, setPreview] = useState<string | null>(null)
    const [result, setResult] = useState<InspectionResult | null>(null)
    const [loading, setLoading] = useState(false)
    const [activeView, setActiveView] = useState<'original' | 'heatmap' | 'mask'>('original')

    const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        const selectedFile = e.target.files?.[0]
        if (selectedFile) {
            setFile(selectedFile)
            setResult(null)
            const reader = new FileReader()
            reader.onload = (e) => setPreview(e.target?.result as string)
            reader.readAsDataURL(selectedFile)
        }
    }, [])

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault()
        const droppedFile = e.dataTransfer.files[0]
        if (droppedFile && droppedFile.type.startsWith('image/')) {
            setFile(droppedFile)
            setResult(null)
            const reader = new FileReader()
            reader.onload = (e) => setPreview(e.target?.result as string)
            reader.readAsDataURL(droppedFile)
        }
    }, [])

    const handleInspect = async () => {
        if (!file) return
        setLoading(true)
        try {
            const data = await inspectImage(file, {
                returnHeatmap: true,
                returnMask: true,
                returnExplanation: true,
            })
            setResult(data)
        } catch (error) {
            console.error('Inspection failed:', error)
        } finally {
            setLoading(false)
        }
    }

    const handleDownload = () => {
        if (result?.heatmap_base64) {
            downloadBase64Image(result.heatmap_base64, `inspection-${result.inspection_id}.png`)
        }
    }

    const getDisplayImage = () => {
        if (activeView === 'heatmap' && result?.heatmap_base64) {
            return result.heatmap_base64
        }
        if (activeView === 'mask' && result?.mask_base64) {
            return result.mask_base64
        }
        return preview
    }

    return (
        <main className="p-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Left: Upload and Preview */}
                <div className="lg:col-span-2 space-y-6">
                    {/* Upload Area */}
                    {!preview && (
                        <Card
                            className="glass-card border-dashed border-2 cursor-pointer hover:border-primary/50 transition-colors"
                            onDrop={handleDrop}
                            onDragOver={(e) => e.preventDefault()}
                        >
                            <CardContent className="p-12">
                                <label className="flex flex-col items-center justify-center cursor-pointer">
                                    <div className="p-4 rounded-full bg-primary/10 mb-4">
                                        <Upload className="w-8 h-8 text-primary" />
                                    </div>
                                    <p className="text-lg font-medium">Drop an image or click to upload</p>
                                    <p className="text-sm text-muted-foreground mt-2">
                                        Supports JPG, PNG, BMP up to 10MB
                                    </p>
                                    <input
                                        type="file"
                                        accept="image/*"
                                        className="hidden"
                                        onChange={handleFileSelect}
                                    />
                                </label>
                            </CardContent>
                        </Card>
                    )}

                    {/* Preview Area */}
                    {preview && (
                        <Card className="glass-card">
                            <CardHeader className="flex flex-row items-center justify-between">
                                <CardTitle className="flex items-center gap-2">
                                    <ImageIcon className="w-5 h-5" />
                                    {file?.name}
                                </CardTitle>
                                <div className="flex items-center gap-2">
                                    {result && (
                                        <Tabs value={activeView} onValueChange={(v) => setActiveView(v as typeof activeView)}>
                                            <TabsList>
                                                <TabsTrigger value="original">Original</TabsTrigger>
                                                <TabsTrigger value="heatmap">Heatmap</TabsTrigger>
                                                <TabsTrigger value="mask">Mask</TabsTrigger>
                                            </TabsList>
                                        </Tabs>
                                    )}
                                    <Button variant="ghost" size="icon">
                                        <ZoomIn className="w-4 h-4" />
                                    </Button>
                                    {result && (
                                        <Button variant="ghost" size="icon" onClick={handleDownload}>
                                            <Download className="w-4 h-4" />
                                        </Button>
                                    )}
                                </div>
                            </CardHeader>
                            <CardContent>
                                <div className="relative bg-muted rounded-lg overflow-hidden">
                                    <img
                                        src={getDisplayImage() || ''}
                                        alt="Inspection preview"
                                        className="w-full h-auto max-h-[500px] object-contain"
                                    />

                                    {/* Overlay bounding boxes */}
                                    {activeView === 'original' && result?.defects.map((defect, idx) => (
                                        <div
                                            key={idx}
                                            className="absolute border-2 pointer-events-none"
                                            style={{
                                                left: `${defect.bounding_box.x_min * 100}%`,
                                                top: `${defect.bounding_box.y_min * 100}%`,
                                                width: `${(defect.bounding_box.x_max - defect.bounding_box.x_min) * 100}%`,
                                                height: `${(defect.bounding_box.y_max - defect.bounding_box.y_min) * 100}%`,
                                                borderColor: defect.severity === 'critical' ? '#ef4444' :
                                                    defect.severity === 'high' ? '#f97316' :
                                                        defect.severity === 'medium' ? '#eab308' : '#22c55e',
                                            }}
                                        >
                                            <span
                                                className="absolute -top-5 left-0 text-xs px-1 rounded text-white"
                                                style={{
                                                    backgroundColor: defect.severity === 'critical' ? '#ef4444' :
                                                        defect.severity === 'high' ? '#f97316' :
                                                            defect.severity === 'medium' ? '#eab308' : '#22c55e',
                                                }}
                                            >
                                                {defect.defect_type}
                                            </span>
                                        </div>
                                    ))}
                                </div>

                                {/* Action Buttons */}
                                <div className="flex gap-3 mt-4">
                                    <Button onClick={handleInspect} disabled={loading} className="flex-1">
                                        {loading ? (
                                            <>
                                                <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                                                Analyzing...
                                            </>
                                        ) : (
                                            <>
                                                <Layers className="w-4 h-4 mr-2" />
                                                Inspect Image
                                            </>
                                        )}
                                    </Button>
                                    <Button
                                        variant="outline"
                                        onClick={() => {
                                            setFile(null)
                                            setPreview(null)
                                            setResult(null)
                                        }}
                                    >
                                        Clear
                                    </Button>
                                </div>
                            </CardContent>
                        </Card>
                    )}
                </div>

                {/* Right: Results */}
                <div className="space-y-6">
                    {/* Status Card */}
                    <Card className={`glass-card ${result?.is_defective ? 'border-severity-high' : 'border-green-500'}`}>
                        <CardContent className="p-6">
                            <div className="flex items-center gap-4">
                                {result?.is_defective ? (
                                    <div className="p-3 rounded-full bg-severity-high/20">
                                        <AlertTriangle className="w-6 h-6 text-severity-high" />
                                    </div>
                                ) : (
                                    <div className="p-3 rounded-full bg-green-500/20">
                                        <CheckCircle className="w-6 h-6 text-green-500" />
                                    </div>
                                )}
                                <div>
                                    <h3 className="text-lg font-semibold">
                                        {result ? (result.is_defective ? 'Defects Detected' : 'No Defects') : 'Awaiting Inspection'}
                                    </h3>
                                    {result && (
                                        <p className="text-sm text-muted-foreground">
                                            {result.defects.length} issue(s) found • {result.processing_time_ms.toFixed(0)}ms
                                        </p>
                                    )}
                                </div>
                            </div>

                            {result && (
                                <div className="mt-4">
                                    <div className="flex justify-between text-sm mb-2">
                                        <span>Anomaly Score</span>
                                        <span className="font-medium">{formatPercent(result.overall_score)}</span>
                                    </div>
                                    <Progress value={result.overall_score * 100} className="h-2" />
                                </div>
                            )}
                        </CardContent>
                    </Card>

                    {/* Defects List */}
                    {result && result.defects.length > 0 && (
                        <Card className="glass-card">
                            <CardHeader>
                                <CardTitle className="text-lg">Detected Defects</CardTitle>
                            </CardHeader>
                            <CardContent className="space-y-3">
                                {result.defects.map((defect, idx) => (
                                    <DefectCard key={idx} defect={defect} />
                                ))}
                            </CardContent>
                        </Card>
                    )}

                    {/* Explanation */}
                    {result?.explanation && (
                        <Card className="glass-card">
                            <CardHeader>
                                <CardTitle className="text-lg">Analysis Summary</CardTitle>
                            </CardHeader>
                            <CardContent>
                                <p className="text-sm text-muted-foreground">{result.explanation}</p>
                            </CardContent>
                        </Card>
                    )}
                </div>
            </div>
        </main>
    )
}

function DefectCard({ defect }: { defect: DefectDetection }) {
    return (
        <div className="p-3 rounded-lg bg-muted/50 border border-border">
            <div className="flex items-center justify-between mb-2">
                <span className="font-medium capitalize">
                    {defect.defect_type.replace('_', ' ')}
                </span>
                <Badge variant={defect.severity as 'critical' | 'high' | 'medium' | 'low' | 'info'}>
                    {defect.severity}
                </Badge>
            </div>
            <div className="grid grid-cols-2 gap-2 text-sm text-muted-foreground">
                <div>Confidence: {formatPercent(defect.confidence)}</div>
                <div>Area: {defect.area_percentage.toFixed(2)}%</div>
            </div>
            <div className="mt-2">
                <div className="flex justify-between text-xs mb-1">
                    <span>Severity Score</span>
                    <span>{formatPercent(defect.severity_score)}</span>
                </div>
                <Progress value={defect.severity_score * 100} className="h-1.5" />
            </div>
        </div>
    )
}
