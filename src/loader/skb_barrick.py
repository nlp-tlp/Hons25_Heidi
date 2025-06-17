from pydantic import Field
import csv

from skb import SKB, SKBSchema, SKBNode

class BarrickSchema(SKBSchema):
    class Source(SKBNode):
        spreadsheet: str = Field(..., id=True)
        index: int = Field(..., id=True)

    class Subsystem(SKBNode):
        name: str = Field(..., id=True)

    class Component(SKBNode):
        in_subsystem: list[str] = Field(..., id=True, relation=True, dest='Subsystem')
        name: str = Field(..., id=True)

    class SubComponent(SKBNode):
        in_component: list[str] = Field(..., id=True, relation=True, dest='Component')
        name: str = Field(..., id=True)

    class FailureMode(SKBNode):
        in_source: list[str] = Field(..., relation=True, dest='Source')
        in_sub_component: list[str] = Field(..., id=True, relation=True, dest='SubComponent')
        has_failure_effect: list[str] = Field(..., relation=True, dest='FailureEffect')
        has_failure_cause: list[str] = Field(..., relation=True, dest='FailureCause')
        has_recommended_action: list[str] = Field(..., relation=True, dest='RecommendedAction')
        has_current_controls: list[str] = Field(..., relation=True, dest='CurrentControls')
        description: str = Field(..., id=True, semantic=True)
        occurrence: int = Field(..., id=True)
        detection: int = Field(..., id=True)
        rpn: int = Field(..., id=True)
        severity: int = Field(..., id=True)

    class FailureEffect(SKBNode):
        description: str = Field(..., id=True, semantic=True)

    class FailureCause(SKBNode):
        description: str = Field(..., id=True, semantic=True)

    class RecommendedAction(SKBNode):
        description: str = Field(..., id=True, semantic=True)

    class CurrentControls(SKBNode):
        description: str = Field(..., id=True, semantic=True)

def load_from_barrick_csv(skb: SKB, filepath: str, max_rows: int = None):
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break

            source = BarrickSchema.Source(spreadsheet=row["Spreadsheet"], index=i)
            source_id = skb.add_entity(source)

            subsystem = BarrickSchema.Subsystem(name=row["Subsystem"])
            subsystem_id = skb.add_entity(subsystem)

            component = BarrickSchema.Component(in_subsystem=[subsystem_id], name=row["Component"])
            component_id = skb.add_entity(component)
            skb.get_entity_by_id(subsystem_id)._rev_in_subsystem = component_id

            subcomponent = BarrickSchema.SubComponent(in_component=[component_id], name=row["Sub-Component"])
            subcomponent_id = skb.add_entity(subcomponent)

            fe = BarrickSchema.FailureEffect(description=row["Potential Effect(s) of Failure"])
            fe_id = skb.add_entity(fe)

            fc = BarrickSchema.FailureCause(description=row["Potential Cause(s) of Failure"])
            fc_id = skb.add_entity(fc)

            controls_str = row["Current Controls"]
            if controls_str:
                controls = BarrickSchema.CurrentControls(description=controls_str)
                controls_id = skb.add_entity(controls)

            recommended_str = row["Recommended Action"]
            if recommended_str:
                recommended = BarrickSchema.RecommendedAction(description=recommended_str)
                recommended_id = skb.add_entity(recommended)

            fm = BarrickSchema.FailureMode(
                in_source=[source_id],
                in_sub_component=[subcomponent_id],
                has_failure_effect=[fe_id],
                has_failure_cause=[fc_id],
                has_recommended_action=[recommended_id] if recommended_id else [],
                has_current_controls=[controls_id] if controls_id else [],
                description=row["Potential Failure Mode"],
                occurrence=int(row["Occurrence"]),
                detection=int(row["Detection"]),
                rpn=int(row["RPN"]),
                severity=int(row["Severity"])
            )
            skb.add_entity(fm)

    return skb

SOURCE_CSV = "fmea_barrick_filled.csv"
STORE_PKL = "skb.pkl"

# Load from CSV
skb = SKB(BarrickSchema)
load_from_barrick_csv(skb, SOURCE_CSV)
entities = skb.get_entities()

# Save to PKL
skb.save_pickle(STORE_PKL)

# Load from PKL
# skb_loaded = SKB(BarrickSchema)
# skb_loaded.load_pickle(STORE_PKL)
# entities_loaded = skb_loaded.get_entities()
