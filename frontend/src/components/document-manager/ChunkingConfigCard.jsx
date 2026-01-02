import PropTypes from "prop-types";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../ui/card";
import { Label } from "../ui/label";
import { RadioGroup, RadioGroupItem } from "../ui/radio-group";
import { Slider } from "../ui/slider";

export function ChunkingConfigCard({
  method,
  onMethodChange,
  chunkSize,
  onChunkSizeChange,
  chunkOverlap,
  onChunkOverlapChange,
  availableMethods = {}
}) {
  const methods = Object.keys(availableMethods).length > 0
    ? availableMethods
    : {
        recursive: { name: "Recursive", description: "Recursively splits at natural boundaries" },
        fixed: { name: "Fixed Size", description: "Splits into fixed-size chunks" },
        semantic: { name: "Semantic", description: "Splits based on semantic meaning" }
      };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">Chunking Configuration</CardTitle>
        <CardDescription>
          Configure how documents are split into chunks for indexing
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Chunking Method */}
        <div className="space-y-3">
          <Label className="text-sm font-medium">Chunking Method</Label>
          <RadioGroup
            value={method}
            onValueChange={onMethodChange}
            className="grid gap-2"
          >
            {Object.entries(methods).map(([key, info]) => (
              <div key={key} className="flex items-center space-x-3">
                <RadioGroupItem value={key} id={`method-${key}`} />
                <Label
                  htmlFor={`method-${key}`}
                  className="flex flex-col cursor-pointer"
                >
                  <span className="font-medium">{info.name || key}</span>
                  <span className="text-xs text-muted-foreground">
                    {info.description || ""}
                  </span>
                </Label>
              </div>
            ))}
          </RadioGroup>
        </div>

        {/* Chunk Size */}
        <div className="space-y-3">
          <div className="flex justify-between">
            <Label className="text-sm font-medium">Chunk Size</Label>
            <span className="text-sm text-muted-foreground">{chunkSize} characters</span>
          </div>
          <Slider
            value={[chunkSize]}
            onValueChange={(value) => onChunkSizeChange(value[0])}
            min={100}
            max={4000}
            step={100}
            className="w-full"
          />
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>100</span>
            <span>4000</span>
          </div>
        </div>

        {/* Chunk Overlap */}
        <div className="space-y-3">
          <div className="flex justify-between">
            <Label className="text-sm font-medium">Chunk Overlap</Label>
            <span className="text-sm text-muted-foreground">{chunkOverlap} characters</span>
          </div>
          <Slider
            value={[chunkOverlap]}
            onValueChange={(value) => onChunkOverlapChange(value[0])}
            min={0}
            max={500}
            step={25}
            className="w-full"
          />
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>0</span>
            <span>500</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

ChunkingConfigCard.propTypes = {
  method: PropTypes.string.isRequired,
  onMethodChange: PropTypes.func.isRequired,
  chunkSize: PropTypes.number.isRequired,
  onChunkSizeChange: PropTypes.func.isRequired,
  chunkOverlap: PropTypes.number.isRequired,
  onChunkOverlapChange: PropTypes.func.isRequired,
  availableMethods: PropTypes.object
};
