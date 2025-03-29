class MorphForm(Enum):
    HUMAN = "Human"
    BIRD = "Bird"
    FISH = "Fish"
    FELINE = "Cat"
    CANINE = "Dog"
    DRAGON = "Dragon"
    ROBOT = "Android"
    ENERGY = "Pure Energy"
    CUSTOM = "Custom"

class AgentMorphology:
    def __init__(self, form=MorphForm.HUMAN, custom_shape: Dict[str, Any] = None):
        self.current_form = form
        self.custom_shape = custom_shape or {}

    def morph_to(self, new_form: MorphForm, custom_shape: Dict[str, Any] = None):
        self.current_form = new_form
        if new_form == MorphForm.CUSTOM:
            self.custom_shape = custom_shape or {}
        logger.info(f"Agent morphed into: {self.describe()}")

    def describe(self):
        if self.current_form == MorphForm.CUSTOM:
            return f"Custom Form: {json.dumps(self.custom_shape)}"
        return f"{self.current_form.value}"
